"""
providers/flightapi.py

FlightAPI.io integration (roundtrip search).
- Each request costs 2 credits
- Response uses cross-referenced IDs for places, carriers, segments, legs
- Deep links go to Skyscanner booking pages
- Switch on with: dokku config:set flyyv FLIGHT_PROVIDER=flightapi
"""

import hashlib
import json as _json
import os
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional

import requests
from fastapi import HTTPException

from schemas.search import CabinClass, CabinSummary, FlightOption, SearchParams

try:
    from airlines import AIRLINE_NAMES
except ImportError:
    AIRLINE_NAMES: Dict[str, str] = {}

FLIGHTAPI_BASE_URL = "https://api.flightapi.io"
SKYSCANNER_BASE = "https://www.skyscanner.com"

# In-memory cache: avoids burning credits on duplicate searches within TTL window
_SEARCH_CACHE: Dict[str, Any] = {}

def _cache_key(origin, destination, dep_str, ret_str, cabin, pax, currency) -> str:
    return f"{origin}|{destination}|{dep_str}|{ret_str}|{cabin}|{pax}|{currency}"

def _cache_get(key: str):
    entry = _SEARCH_CACHE.get(key)
    if not entry:
        return None
    if datetime.utcnow() > entry["expires_at"]:
        del _SEARCH_CACHE[key]
        return None
    print(f"[flightapi] cache HIT key={key}")
    return entry["results"]

def _cache_set(key: str, results: list) -> None:
    ttl = int(os.getenv("FLIGHTAPI_CACHE_TTL_SECONDS", "1800"))
    _SEARCH_CACHE[key] = {"results": results, "expires_at": datetime.utcnow() + timedelta(seconds=ttl)}



def _get_api_key() -> str:
    key = os.getenv("FLIGHTAPI_KEY", "").strip()
    if not key:
        raise HTTPException(status_code=500, detail="FLIGHTAPI_KEY is not configured")
    return key


def _map_cabin(cabin: Optional[str]) -> str:
    if not cabin:
        return "Business"
    cabin_upper = cabin.upper()
    if cabin_upper in ("BUSINESS", "B"):
        return "Business"
    if cabin_upper in ("FIRST", "F"):
        return "First"
    if cabin_upper in ("PREMIUM_ECONOMY", "W"):
        return "Premium_Economy"
    return "Economy"


def _build_lookup_maps(data: dict) -> tuple:
    places_map: Dict[Any, Dict] = {}
    for p in data.get("places", []):
        pid = p.get("id") or p.get("entityId")
        if pid is None:
            continue
        places_map[pid] = {
            "iata": p.get("iata") or p.get("displayCode") or "",
            "name": p.get("name") or "",
        }

    carriers_map: Dict[Any, Dict] = {}
    for c in data.get("carriers", []):
        cid = c.get("id")
        if cid is None:
            continue
        code = c.get("iata") or c.get("displayCode") or c.get("code") or ""
        carriers_map[cid] = {
            "code": code,
            "name": c.get("name") or AIRLINE_NAMES.get(code, code),
        }

    segments_map: Dict[str, Dict] = {}
    for s in data.get("segments", []):
        sid = s.get("id")
        if sid:
            segments_map[sid] = s

    return places_map, carriers_map, segments_map


def _build_legs_map(data: dict) -> Dict[str, Dict]:
    legs_map: Dict[str, Dict] = {}
    for leg in data.get("legs", []):
        lid = leg.get("id")
        if lid:
            legs_map[lid] = leg
    return legs_map


def _place_iata(place_id: Any, places_map: Dict) -> str:
    return (places_map.get(place_id) or {}).get("iata") or ""


def _place_name(place_id: Any, places_map: Dict) -> str:
    return (places_map.get(place_id) or {}).get("name") or ""


def _carrier_code(carrier_id: Any, carriers_map: Dict) -> str:
    return (carriers_map.get(carrier_id) or {}).get("code") or ""


def _carrier_name(carrier_id: Any, carriers_map: Dict) -> str:
    return (carriers_map.get(carrier_id) or {}).get("name") or ""


def _parse_dt(dt_str: Optional[str]) -> Optional[datetime]:
    if not dt_str:
        return None
    for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(dt_str, fmt)
        except Exception:
            continue
    return None


def _build_segments_for_leg(leg, segments_map, places_map, carriers_map, direction):
    out = []
    seg_ids = leg.get("segment_ids") or []
    for idx, sid in enumerate(seg_ids):
        seg = segments_map.get(sid)
        if not seg:
            continue

        orig_id = seg.get("origin_place_id")
        dest_id = seg.get("destination_place_id")
        carrier_id = seg.get("marketing_carrier_id")
        op_carrier_id = seg.get("operating_carrier_id")

        dep_at = seg.get("departure") or ""
        arr_at = seg.get("arrival") or ""
        arr_dt = _parse_dt(arr_at)

        layover_mins = None
        if idx < len(seg_ids) - 1:
            next_seg = segments_map.get(seg_ids[idx + 1])
            if next_seg:
                next_dep = _parse_dt(next_seg.get("departure") or "")
                if arr_dt and next_dep:
                    try:
                        layover_mins = max(0, int((next_dep - arr_dt).total_seconds() // 60))
                    except Exception:
                        pass

        carrier_code = _carrier_code(carrier_id, carriers_map)
        op_code = _carrier_code(op_carrier_id, carriers_map) if op_carrier_id != carrier_id else None

        out.append({
            "direction": direction,
            "flightNumber": seg.get("marketing_flight_number") or "",
            "marketingCarrier": carrier_code,
            "operatingCarrier": op_code,
            "origin": _place_iata(orig_id, places_map),
            "destination": _place_iata(dest_id, places_map),
            "originAirport": _place_name(orig_id, places_map),
            "destinationAirport": _place_name(dest_id, places_map),
            "departingAt": dep_at,
            "arrivingAt": arr_at,
            "durationMinutes": seg.get("duration"),
            "layoverMinutesToNext": layover_mins,
            "cabin": None,
            "bookingCode": None,
            "fareBrand": None,
        })
    return out


def _map_itinerary_to_option(itin, legs_map, segments_map, places_map, carriers_map, passengers, currency, dep_date, ret_date):
    cheapest = itin.get("cheapest_price") or {}
    price_total = cheapest.get("amount")
    if price_total is None:
        for po in itin.get("pricing_options") or []:
            price_total = (po.get("price") or {}).get("amount")
            if price_total is not None:
                break
    if price_total is None:
        return None
    try:
        price_total = float(price_total)
    except Exception:
        return None

    pax = max(1, int(passengers or 1))
    price_per_pax = round(price_total / pax, 2)

    deep_link = None
    for po in itin.get("pricing_options") or []:
        for item in po.get("items") or []:
            url = item.get("url") or ""
            if url:
                deep_link = SKYSCANNER_BASE + url if url.startswith("/") else url
                break
        if deep_link:
            break

    leg_ids = itin.get("leg_ids") or []
    outbound_leg = legs_map.get(leg_ids[0]) if len(leg_ids) > 0 else None
    return_leg = legs_map.get(leg_ids[1]) if len(leg_ids) > 1 else None

    if not outbound_leg:
        return None

    outbound_segs = _build_segments_for_leg(outbound_leg, segments_map, places_map, carriers_map, "outbound")
    return_segs = _build_segments_for_leg(return_leg, segments_map, places_map, carriers_map, "return") if return_leg else []

    outbound_duration = outbound_leg.get("duration") or 0
    return_duration = (return_leg.get("duration") or 0) if return_leg else 0
    total_duration = outbound_duration + return_duration

    def iso8601_duration(minutes):
        mins = max(0, int(minutes or 0))
        h, m = divmod(mins, 60)
        if h and m:
            return f"PT{h}H{m}M"
        if h:
            return f"PT{h}H"
        return f"PT{m}M"

    stops_outbound = outbound_leg.get("stop_count") or max(0, len(outbound_segs) - 1)

    stopover_codes = []
    stopover_airports = []
    if stops_outbound > 0 and outbound_segs:
        for seg in outbound_segs[:-1]:
            if seg.get("destination"):
                stopover_codes.append(seg["destination"])
            if seg.get("destinationAirport"):
                stopover_airports.append(seg["destinationAirport"])

    carrier_ids = outbound_leg.get("marketing_carrier_ids") or []
    top_carrier_id = carrier_ids[0] if carrier_ids else None
    top_carrier_code = _carrier_code(top_carrier_id, carriers_map) if top_carrier_id is not None else ""
    top_carrier_name = _carrier_name(top_carrier_id, carriers_map) if top_carrier_id is not None else ""
    if not top_carrier_name and top_carrier_code:
        top_carrier_name = AIRLINE_NAMES.get(top_carrier_code, top_carrier_code)

    origin_code = _place_iata(outbound_leg.get("origin_place_id"), places_map)
    destination_code = _place_iata(outbound_leg.get("destination_place_id"), places_map)
    origin_airport = _place_name(outbound_leg.get("origin_place_id"), places_map)
    destination_airport = _place_name(outbound_leg.get("destination_place_id"), places_map)

    id_src = _json.dumps({
        "itin_id": itin.get("id") or "",
        "dep": dep_date.isoformat(),
        "ret": ret_date.isoformat(),
        "price": str(price_per_pax),
        "carrier": top_carrier_code,
    }, sort_keys=True)
    offer_id = hashlib.md5(id_src.encode()).hexdigest()

    return FlightOption(
        id=offer_id,
        provider="flightapi",
        providerSessionId=None,
        providerRecommendationId=itin.get("id"),
        airline=top_carrier_name or top_carrier_code or "Airline",
        airlineCode=top_carrier_code or None,
        price=price_per_pax,
        currency=currency,
        departureDate=dep_date.isoformat(),
        returnDate=ret_date.isoformat(),
        stops=stops_outbound,
        cabinSummary=CabinSummary.BUSINESS,
        cabinHighest=CabinClass.BUSINESS,
        cabinByDirection=None,
        durationMinutes=outbound_duration,
        totalDurationMinutes=total_duration,
        duration=iso8601_duration(total_duration),
        origin=origin_code,
        destination=destination_code,
        originAirport=origin_airport,
        destinationAirport=destination_airport,
        stopoverCodes=stopover_codes or None,
        stopoverAirports=stopover_airports or None,
        outboundSegments=outbound_segs or None,
        returnSegments=return_segs or None,
        aircraftCodes=None,
        aircraftNames=None,
        bookingUrl=deep_link,
        url=deep_link,
    )


def run_flightapi_scan(
    params: SearchParams,
    dep_override: Optional[date] = None,
    ret_override: Optional[date] = None,
) -> List[FlightOption]:
    """
    Execute a roundtrip search via FlightAPI.io.
    Costs 2 credits per call.
    """
    api_key = _get_api_key()

    dep = dep_override or getattr(params, "earliestDeparture", None)
    ret = ret_override

    if ret is None:
        try:
            nights = getattr(params, "minStayDays", None) or getattr(params, "nights", None)
            if nights and dep:
                ret = dep + timedelta(days=int(nights))
        except Exception:
            pass

    if not dep or not ret:
        print("[flightapi] missing dep or ret date, skipping")
        return []

    origin = getattr(params, "origin", None)
    destination = getattr(params, "destination", None)
    if not origin or not destination:
        print("[flightapi] missing origin or destination, skipping")
        return []

    pax = int(getattr(params, "passengers", 1) or 1)
    cabin = _map_cabin(getattr(params, "cabin", "Business"))
    currency = os.getenv("FLIGHTAPI_CURRENCY", "USD")

    dep_str = dep.strftime("%Y-%m-%d")
    ret_str = ret.strftime("%Y-%m-%d")

    # Check cache first to avoid burning credits on repeated identical searches
    ck = _cache_key(origin, destination, dep_str, ret_str, cabin, pax, currency)
    cached = _cache_get(ck)
    if cached is not None:
        # Re-map from raw data so UI/mapping changes are always reflected
        raw_data = cached.get("raw")
        if raw_data:
            print(f"[flightapi] cache HIT — re-mapping from raw data key={ck}")
            places_map, carriers_map, segments_map = _build_lookup_maps(raw_data)
            legs_map = _build_legs_map(raw_data)
            itineraries = raw_data.get("itineraries") or []
            max_results = int(os.getenv("MAX_OFFERS_PER_PAIR", "20"))
            remapped = []
            for itin in itineraries:
                if len(remapped) >= max_results:
                    break
                try:
                    opt = _map_itinerary_to_option(
                        itin=itin,
                        legs_map=legs_map,
                        segments_map=segments_map,
                        places_map=places_map,
                        carriers_map=carriers_map,
                        passengers=pax,
                        currency=currency,
                        dep_date=dep,
                        ret_date=ret,
                    )
                    if opt:
                        remapped.append(opt)
                except Exception as e:
                    continue
            return remapped

    url = (
        f"{FLIGHTAPI_BASE_URL}/roundtrip"
        f"/{api_key}/{origin}/{destination}"
        f"/{dep_str}/{ret_str}"
        f"/{pax}/0/0/{cabin}/{currency}"
    )

    print(f"[flightapi] GET roundtrip origin={origin} dest={destination} dep={dep_str} ret={ret_str} cabin={cabin} pax={pax} currency={currency}")

    try:
        resp = requests.get(url, timeout=45)
    except Exception as e:
        print(f"[flightapi] request failed: {e}")
        return []

    if resp.status_code == 429:
        print("[flightapi] *** CREDIT LIMIT HIT (429) — upgrade your plan at api.flightapi.io ***")
        return []

    if resp.status_code >= 400:
        print(f"[flightapi] error {resp.status_code}: {resp.text[:500]}")
        return []

    try:
        data = resp.json()
    except Exception as e:
        print(f"[flightapi] JSON parse failed: {e}")
        return []

    if not isinstance(data, dict):
        print(f"[flightapi] unexpected response type: {type(data)}")
        return []

    itineraries = data.get("itineraries") or []
    print(f"[flightapi] itineraries={len(itineraries)} legs={len(data.get('legs', []))} segments={len(data.get('segments', []))} places={len(data.get('places', []))} carriers={len(data.get('carriers', []))}")

    if not itineraries:
        print("[flightapi] no itineraries returned")
        return []

    places_map, carriers_map, segments_map = _build_lookup_maps(data)
    legs_map = _build_legs_map(data)

    mapped: List[FlightOption] = []
    max_results = int(os.getenv("MAX_OFFERS_PER_PAIR", "20"))

    for itin in itineraries:
        if len(mapped) >= max_results:
            break
        try:
            opt = _map_itinerary_to_option(
                itin=itin,
                legs_map=legs_map,
                segments_map=segments_map,
                places_map=places_map,
                carriers_map=carriers_map,
                passengers=pax,
                currency=currency,
                dep_date=dep,
                ret_date=ret,
            )
            if opt:
                mapped.append(opt)
        except Exception as e:
            print(f"[flightapi] map_failed: {type(e).__name__}: {e}")
            continue

    if mapped:
        o0 = mapped[0]
        print(f"[flightapi] mapped={len(mapped)} first_airline={o0.airline} first_price={o0.price} {o0.currency} deeplink={'YES' if o0.url else 'NO'}")
    else:
        print("[flightapi] mapped=0")

    # Cache the raw API data, not the mapped results
    # This means UI/mapping changes are always reflected without clearing cache
    _cache_set(ck, {"raw": data})

    return mapped