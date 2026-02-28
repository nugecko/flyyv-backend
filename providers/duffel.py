"""
providers/duffel.py

Duffel API helpers:
- Low-level HTTP wrappers (duffel_get, duffel_post)
- Offer request creation
- Offer listing
- Offer-to-FlightOption mapping
- Direct-only offer fetching
- Full scan (legacy alert fallback path)

NOTE: Duffel is currently used only for direct-only fetches and as a legacy
fallback in alert scanning. TTN is the primary search provider for the
/search-business route.
"""

import os
import re
import time
from collections import Counter, defaultdict
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import requests
from fastapi import HTTPException

from config import (
    DUFFEL_API_BASE,
    MAX_DATE_PAIRS_HARD,
    MAX_OFFERS_PER_PAIR_HARD,
    MAX_OFFERS_TOTAL_HARD,
    get_config_int,
    get_config_str,
)
from schemas.search import CabinClass, CabinSummary, FlightOption, SearchParams

try:
    from airlines import AIRLINE_NAMES, AIRLINE_BOOKING_URLS as AIRLINE_BOOKING_URL
except ImportError:
    AIRLINE_NAMES: Dict[str, str] = {}
    AIRLINE_BOOKING_URL: Dict[str, str] = {}


# =====================================================================
# SECTION: DURATION HELPERS
# =====================================================================

def iso8601_duration(minutes: int) -> str:
    mins = max(0, int(minutes or 0))
    hours = mins // 60
    mins = mins % 60
    if hours and mins:
        return f"PT{hours}H{mins}M"
    if hours:
        return f"PT{hours}H"
    return f"PT{mins}M"


def parse_iso8601_duration(duration_str: Optional[str]) -> Optional[int]:
    """Parse ISO 8601 duration string (e.g., 'PT9H30M') to minutes."""
    if not duration_str:
        return None
    match = re.match(r"^PT(?:(\d+)H)?(?:(\d+)M)?$", duration_str)
    if not match:
        return None
    hours = int(match.group(1) or 0)
    minutes = int(match.group(2) or 0)
    return hours * 60 + minutes


# =====================================================================
# SECTION: CABIN HELPERS
# =====================================================================

CABIN_RANK = {
    "ECONOMY": 1,
    "PREMIUM_ECONOMY": 2,
    "BUSINESS": 3,
    "FIRST": 4,
}


def normalize_cabin(raw: Optional[str]) -> Optional[CabinClass]:
    if not raw:
        return None
    val = str(raw).strip().upper().replace(" ", "_")
    if val in ("ECONOMY", "PREMIUM_ECONOMY", "BUSINESS", "FIRST"):
        return CabinClass(val)
    return None


def extract_segment_cabin(seg: dict) -> Optional[CabinClass]:
    passengers = seg.get("passengers") or []
    if isinstance(passengers, list) and passengers:
        p0 = passengers[0] or {}
        cabin = p0.get("cabin_class") or p0.get("cabin") or p0.get("cabinClass")
        norm = normalize_cabin(cabin)
        if norm:
            return norm
    cabin = seg.get("cabin_class") or seg.get("cabin") or seg.get("cabinClass")
    return normalize_cabin(cabin)


def extract_segment_booking_code(seg: dict) -> Optional[str]:
    passengers = seg.get("passengers") or []
    if isinstance(passengers, list) and passengers:
        p0 = passengers[0] or {}
        code = p0.get("booking_code") or p0.get("bookingCode")
        if code:
            return str(code)
    code = seg.get("booking_code") or seg.get("bookingCode")
    return str(code) if code else None


def extract_segment_fare_brand(seg: dict) -> Optional[str]:
    passengers = seg.get("passengers") or []
    if isinstance(passengers, list) and passengers:
        p0 = passengers[0] or {}
        brand = p0.get("fare_brand_name") or p0.get("fareBrand") or p0.get("fare_brand")
        if brand:
            return str(brand)
    brand = seg.get("fare_brand_name") or seg.get("fareBrand") or seg.get("fare_brand")
    return str(brand) if brand else None


def summarize_cabins(cabins: List[Optional[CabinClass]]) -> Tuple[CabinSummary, Optional[CabinClass]]:
    if any(c is None for c in cabins) or not cabins:
        return CabinSummary.UNKNOWN, None
    unique = {c.value for c in cabins if c is not None}
    if len(unique) == 1:
        single = cabins[0]
        return CabinSummary(single.value), single
    highest = max((c for c in cabins if c is not None), key=lambda c: CABIN_RANK.get(c.value, 0))
    return CabinSummary.MIXED, highest


# =====================================================================
# SECTION: LOW LEVEL HTTP HELPERS
# =====================================================================

def _duffel_token() -> str:
    token = (os.getenv("DUFFEL_API_TOKEN") or os.getenv("DUFFEL_ACCESS_TOKEN") or "").strip()
    if not token:
        raise HTTPException(status_code=500, detail="Duffel token is not configured")
    return token


def duffel_post(path: str, payload: dict) -> dict:
    token = _duffel_token()
    url = "https://api.duffel.com" + path
    headers = {
        "Authorization": f"Bearer {token}",
        "Duffel-Version": "v2",
        "Content-Type": "application/json",
    }
    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=45)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Duffel request failed: {e}")

    try:
        data = resp.json()
    except Exception:
        data = {"raw": resp.text}

    try:
        request_id = (
            resp.headers.get("Request-Id")
            or resp.headers.get("Duffel-Request-Id")
            or resp.headers.get("X-Request-Id")
        )
        safe_body = (resp.text or "").replace("\n", "\\n").replace("\r", "\\r")
        if resp.status_code >= 400:
            print(f"Duffel POST {path} status={resp.status_code} request_id={request_id} body={safe_body[:2000]}")
        else:
            print(f"Duffel POST {path} status={resp.status_code} request_id={request_id}")
    except Exception:
        pass

    if resp.status_code >= 400:
        raise HTTPException(status_code=resp.status_code, detail=data)

    if isinstance(data, dict) and "data" in data:
        return data["data"]
    return data


def duffel_get(path: str, params: Optional[dict] = None) -> dict:
    token = _duffel_token()
    url = "https://api.duffel.com" + path
    headers = {
        "Authorization": f"Bearer {token}",
        "Duffel-Version": "v2",
        "Content-Type": "application/json",
    }
    try:
        resp = requests.get(url, headers=headers, params=params or {}, timeout=45)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Duffel request failed: {e}")

    try:
        data = resp.json()
    except Exception:
        data = {"raw": resp.text}

    try:
        request_id = (
            resp.headers.get("Request-Id")
            or resp.headers.get("Duffel-Request-Id")
            or resp.headers.get("X-Request-Id")
        )
        if resp.status_code >= 400:
            safe_body = (resp.text or "").replace("\n", "\\n").replace("\r", "\\r")
            print(f"Duffel GET {path} status={resp.status_code} request_id={request_id} body={safe_body[:1200]}")
        else:
            print(f"Duffel GET {path} status={resp.status_code} request_id={request_id}")
    except Exception:
        pass

    if resp.status_code >= 400:
        raise HTTPException(status_code=resp.status_code, detail=data)

    if isinstance(data, dict) and "data" in data:
        return data["data"]
    return data


def duffel_list_offers(offer_request_id: str, limit: int = 100) -> List[dict]:
    res = duffel_get("/air/offers", params={"offer_request_id": offer_request_id, "limit": int(limit)})
    if isinstance(res, list):
        return res
    if isinstance(res, dict) and "data" in res and isinstance(res["data"], list):
        return res["data"]
    return []


def duffel_create_offer_request(slices: List[dict], passengers: List[dict], cabin: str) -> dict:
    cabin_val = (cabin or "").strip().upper().replace(" ", "_")
    cabin_map = {
        "ECONOMY": "economy",
        "PREMIUM_ECONOMY": "premium_economy",
        "BUSINESS": "business",
        "FIRST": "first",
    }
    duffel_cabin = cabin_map.get(cabin_val)

    payload: Dict[str, Any] = {
        "data": {
            "slices": slices,
            "passengers": passengers,
        }
    }
    if duffel_cabin:
        payload["data"]["cabin_class"] = duffel_cabin

    return duffel_post("/air/offer_requests", payload)


# =====================================================================
# SECTION: OFFER MAPPING
# =====================================================================

def map_duffel_offer_to_option(
    offer: dict,
    dep: date,
    ret: date,
    passengers: int,
) -> FlightOption:
    """
    PRICE CONTRACT:
    - Duffel offer.total_amount is TOTAL for all passengers
    - FlightOption.price is PER PASSENGER
    """
    pax = max(1, int(passengers or 1))

    total_price = float(offer.get("total_amount", 0) or 0)
    price = total_price / pax
    currency = offer.get("total_currency", "GBP")

    owner = offer.get("owner", {}) or {}
    airline_code = owner.get("iata_code")
    airline_name = AIRLINE_NAMES.get(airline_code, owner.get("name", airline_code or "Airline"))
    booking_url = AIRLINE_BOOKING_URL.get(airline_code) if isinstance(AIRLINE_BOOKING_URL, dict) else None

    slices = offer.get("slices", []) or []
    outbound_segments_json: List[dict] = []
    return_segments_json: List[dict] = []

    if len(slices) >= 1:
        outbound_segments_json = slices[0].get("segments", []) or []
    if len(slices) >= 2:
        return_segments_json = slices[1].get("segments", []) or []

    stops_outbound = max(0, len(outbound_segments_json) - 1)

    origin_code = None
    destination_code = None
    origin_airport = None
    destination_airport = None

    if outbound_segments_json:
        first_segment = outbound_segments_json[0]
        last_segment = outbound_segments_json[-1]
        origin_obj = first_segment.get("origin", {}) or {}
        dest_obj = last_segment.get("destination", {}) or {}
        origin_code = origin_obj.get("iata_code")
        destination_code = dest_obj.get("iata_code")
        origin_airport = origin_obj.get("name")
        destination_airport = dest_obj.get("name")

    outbound_segments_info: List[Dict[str, Any]] = []
    return_segments_info: List[Dict[str, Any]] = []
    aircraft_codes: List[str] = []
    aircraft_names: List[str] = []
    outbound_total_minutes = 0
    return_total_minutes = 0

    def parse_iso(dt_str: Optional[str]) -> Optional[datetime]:
        if not dt_str:
            return None
        try:
            return datetime.fromisoformat(dt_str.replace("Z", "+00:00"))
        except Exception:
            return None

    def process_segment_list(direction: str, seg_list: List[dict]) -> Tuple[List[Dict[str, Any]], int]:
        result: List[Dict[str, Any]] = []
        total_minutes = 0

        for idx, seg in enumerate(seg_list):
            o = seg.get("origin", {}) or {}
            d = seg.get("destination", {}) or {}
            aircraft = seg.get("aircraft", {}) or {}

            aircraft_code = aircraft.get("iata_code")
            aircraft_name = aircraft.get("name")

            if aircraft_code:
                aircraft_codes.append(aircraft_code)
            if aircraft_name:
                aircraft_names.append(aircraft_name)

            dep_at_str = seg.get("departing_at")
            arr_at_str = seg.get("arriving_at")
            dep_dt = parse_iso(dep_at_str)
            arr_dt = parse_iso(arr_at_str)

            duffel_duration = seg.get("duration")
            duration_minutes_seg = parse_iso8601_duration(duffel_duration)

            layover_minutes_to_next: Optional[int] = None
            if idx < len(seg_list) - 1:
                this_arr_str = seg.get("arriving_at")
                next_dep_str = seg_list[idx + 1].get("departing_at")
                this_arr_dt = parse_iso(this_arr_str)
                next_dep_dt = parse_iso(next_dep_str)
                if this_arr_dt and next_dep_dt:
                    try:
                        layover_minutes_to_next = int((next_dep_dt - this_arr_dt).total_seconds() // 60)
                    except Exception:
                        layover_minutes_to_next = None

            if duration_minutes_seg:
                total_minutes += duration_minutes_seg
            if layover_minutes_to_next:
                total_minutes += layover_minutes_to_next

            seg_cabin = extract_segment_cabin(seg)

            result.append({
                "direction": direction,
                "flightNumber": seg.get("marketing_carrier_flight_number"),
                "marketingCarrier": (seg.get("marketing_carrier") or {}).get("iata_code"),
                "operatingCarrier": (seg.get("operating_carrier") or {}).get("iata_code"),
                "origin": o.get("iata_code"),
                "destination": d.get("iata_code"),
                "originAirport": o.get("name"),
                "destinationAirport": d.get("name"),
                "departingAt": dep_at_str,
                "arrivingAt": arr_at_str,
                "aircraftCode": aircraft_code,
                "aircraftName": aircraft_name,
                "durationMinutes": duration_minutes_seg,
                "layoverMinutesToNext": layover_minutes_to_next,
                "cabin": (seg_cabin.value if seg_cabin else None),
                "bookingCode": extract_segment_booking_code(seg),
                "fareBrand": extract_segment_fare_brand(seg),
            })

        return result, total_minutes

    outbound_segments_info, outbound_total_minutes = process_segment_list("outbound", outbound_segments_json)
    return_segments_info, return_total_minutes = process_segment_list("return", return_segments_json)

    outbound_cabins = [normalize_cabin(seg.get("cabin")) for seg in outbound_segments_info] if outbound_segments_info else []
    return_cabins = [normalize_cabin(seg.get("cabin")) for seg in return_segments_info] if return_segments_info else []
    all_cabins = outbound_cabins + return_cabins

    cabin_summary, cabin_highest = summarize_cabins(all_cabins)

    cabin_by_direction: Optional[Dict[str, Optional[CabinSummary]]] = None
    if outbound_segments_info or return_segments_info:
        outbound_summary, _ = summarize_cabins(outbound_cabins) if outbound_segments_info else (CabinSummary.UNKNOWN, None)
        return_summary, _ = summarize_cabins(return_cabins) if return_segments_info else (CabinSummary.UNKNOWN, None)
        cabin_by_direction = {"outbound": outbound_summary, "return": return_summary}

    duration_minutes = outbound_total_minutes
    total_duration_minutes = outbound_total_minutes + return_total_minutes
    iso_dur = iso8601_duration(total_duration_minutes)

    stopover_codes: List[str] = []
    stopover_airports: List[str] = []

    if stops_outbound > 0 and outbound_segments_json:
        for seg in outbound_segments_json[:-1]:
            dest = (seg.get("destination") or {}).get("iata_code")
            dest_name = (seg.get("destination") or {}).get("name")
            if dest:
                stopover_codes.append(dest)
            if dest_name:
                stopover_airports.append(dest_name)

    return FlightOption(
        id=offer.get("id", ""),
        airline=airline_name,
        airlineCode=airline_code or None,
        price=price,
        currency=currency,
        departureDate=dep.isoformat(),
        returnDate=ret.isoformat(),
        stops=stops_outbound,
        cabinSummary=cabin_summary,
        cabinHighest=cabin_highest,
        cabinByDirection=cabin_by_direction,
        durationMinutes=duration_minutes,
        totalDurationMinutes=total_duration_minutes,
        duration=iso_dur,
        origin=origin_code,
        destination=destination_code,
        originAirport=origin_airport,
        destinationAirport=destination_airport,
        stopoverCodes=stopover_codes or None,
        stopoverAirports=stopover_airports or None,
        outboundSegments=outbound_segments_info or None,
        returnSegments=return_segments_info or None,
        aircraftCodes=aircraft_codes or None,
        aircraftNames=aircraft_names or None,
        bookingUrl=booking_url,
        url=booking_url,
    )


# =====================================================================
# SECTION: DIRECT ONLY FETCH
# =====================================================================

def fetch_direct_only_offers(
    origin: str,
    destination: str,
    dep_date: date,
    ret_date: date,
    passengers: int,
    cabin: str,
    per_pair_limit: int = 15,
) -> List[FlightOption]:
    if not (os.getenv("DUFFEL_API_TOKEN") or os.getenv("DUFFEL_ACCESS_TOKEN")):
        print("[direct_only] Duffel not configured")
        return []

    slices = [
        {"origin": origin, "destination": destination, "departure_date": dep_date.isoformat()},
        {"origin": destination, "destination": origin, "departure_date": ret_date.isoformat()},
    ]

    # Micro-step: force city origins to airport IATA for Duffel
    for s in slices:
        o = s.get("origin")
        if isinstance(o, dict) and o.get("type") == "city":
            airports = o.get("airports") or []
            if airports:
                preferred = next((a for a in airports if a.get("iata_code") == "LHR"), None)
                chosen = preferred or airports[0]
                s["origin"] = {"iata_code": chosen.get("iata_code")}
                s["origin_type"] = "airport"

    pax = [{"type": "adult"} for _ in range(passengers)]

    payload = {
        "data": {
            "slices": slices,
            "passengers": pax,
            "cabin_class": cabin.lower().replace(" ", "_"),
            "max_connections": 0,
        }
    }

    try:
        data = duffel_post("/air/offer_requests", payload)
    except Exception as e:
        print(f"[direct_only] error creating request: {e}")
        return []

    offer_request_id = data.get("id")
    if not offer_request_id:
        print("[direct_only] no offer_request_id returned")
        return []

    try:
        offers_json = duffel_list_offers(offer_request_id, limit=per_pair_limit)
        if not offers_json:
            time.sleep(1.5)
            offers_json = duffel_list_offers(offer_request_id, limit=per_pair_limit)
    except Exception as e:
        print(f"[direct_only] error listing offers: {e}")
        return []

    results: List[FlightOption] = []
    for offer in offers_json:
        try:
            opt = map_duffel_offer_to_option(offer, dep_date, ret_date, passengers=passengers)
            results.append(opt)
        except Exception as e:
            print(f"[direct_only] mapping error: {e}")

    print(f"[direct_only] fetched {len(results)} direct offers")
    return results


# =====================================================================
# SECTION: SKYSCANNER URL BUILDER
# =====================================================================

def build_skyscanner_url(
    origin: str,
    destination: str,
    dep_date: date,
    ret_date: date,
    cabin: str,
    passengers: int,
) -> str:
    """
    Build a Skyscanner search URL from route parameters.
    Opens a pre-filtered Business Class search for the exact route and dates.
    """
    cabin_map = {
        "BUSINESS": "business",
        "FIRST": "first",
        "PREMIUM_ECONOMY": "premiumeconomy",
        "ECONOMY": "economy",
    }
    cabin_val = cabin_map.get((cabin or "").upper().replace(" ", "_"), "business")
    dep_str = dep_date.strftime("%Y%m%d") if dep_date else ""
    ret_str = ret_date.strftime("%Y%m%d") if ret_date else ""
    pax = max(1, int(passengers or 1))

    return (
        f"https://www.skyscanner.com/transport/flights"
        f"/{origin}/{destination}/{dep_str}/{ret_str}"
        f"/?adults={pax}&cabinclass={cabin_val}&ref=flyyv"
    )


# =====================================================================
# SECTION: FULL SCAN (PRIMARY SEARCH PROVIDER)
# =====================================================================

def run_duffel_scan(
    params: SearchParams,
    dep_override: Optional[date] = None,
    ret_override: Optional[date] = None,
) -> List[FlightOption]:
    """
    Full Duffel scan for a single date pair.
    Results get Skyscanner booking URLs (not direct Duffel booking).
    This is the primary search path when FLIGHT_PROVIDER=duffel.
    """
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
        print("[duffel] missing dep or ret date, skipping")
        return []

    origin = getattr(params, "origin", None)
    destination = getattr(params, "destination", None)
    if not origin or not destination:
        print("[duffel] missing origin or destination, skipping")
        return []

    cabin_raw = getattr(params, "cabin", "BUSINESS") or "BUSINESS"
    cabin = (cabin_raw or "").strip().upper().replace(" ", "_")
    passengers = int(getattr(params, "passengers", 1) or 1)
    max_connections = getattr(params, "maxConnections", None)

    slices = [
        {"origin": origin, "destination": destination, "departure_date": dep.isoformat()},
        {"origin": destination, "destination": origin, "departure_date": ret.isoformat()},
    ]
    pax = [{"type": "adult"} for _ in range(passengers)]

    payload: Dict[str, Any] = {
        "data": {
            "slices": slices,
            "passengers": pax,
            "cabin_class": cabin.lower(),  # Duffel expects: economy, premium_economy, business, first
        }
    }
    if max_connections is not None:
        payload["data"]["max_connections"] = int(max_connections)

    print(f"[duffel] scan origin={origin} dest={destination} dep={dep} ret={ret} cabin={cabin} pax={passengers}")

    try:
        data = duffel_post("/air/offer_requests", payload)
    except Exception as e:
        print(f"[duffel] offer_request failed: {e}")
        return []

    offer_request_id = data.get("id")
    if not offer_request_id:
        print("[duffel] no offer_request_id")
        return []

    try:
        offers_json = duffel_list_offers(offer_request_id, limit=50)
        if not offers_json:
            time.sleep(1.5)
            offers_json = duffel_list_offers(offer_request_id, limit=50)
    except Exception as e:
        print(f"[duffel] list_offers failed: {e}")
        return []

    print(f"[duffel] offers={len(offers_json)}")

    # Import booking URL builder
    try:
        from airline_search_urls import get_booking_urls
    except ImportError:
        get_booking_urls = None

    results: List[FlightOption] = []
    for offer in offers_json:
        try:
            opt = map_duffel_offer_to_option(offer, dep, ret, passengers=passengers)

            # Construct test Duffel operator URL (your dashboard - for testing only)
            offer_id = offer.get("id", "")
            duffel_test_url = f"https://app.duffel.com/flyyv/live/search-v2/{offer_request_id}/{offer_id}" if offer_request_id and offer_id else None

            # Build all three booking URLs for this specific result
            if get_booking_urls:
                booking_data = get_booking_urls(
                    airline_code=opt.airlineCode or "",
                    origin=origin,
                    destination=destination,
                    dep_date=dep,
                    ret_date=ret,
                    cabin=cabin,
                    passengers=passengers,
                    price=opt.price,
                    currency=opt.currency,
                )
                opt = opt.model_copy(update={
                    "url": booking_data["skyscanner"],
                    "bookingUrl": booking_data["airline"] or booking_data["skyscanner"],
                    "bookingUrls": booking_data,
                    "offerRequestId": offer_request_id,
                    "duffelLink": duffel_test_url,
                })
            else:
                sky_url = build_skyscanner_url(origin, destination, dep, ret, cabin, passengers)
                opt = opt.model_copy(update={"bookingUrl": sky_url, "url": sky_url})

            results.append(opt)
        except Exception as e:
            print(f"[duffel] map error: {e}")
            continue

    if results:
        r0 = results[0]
        print(f"[duffel] mapped={len(results)} first_airline={r0.airline} first_price={r0.price} {r0.currency} url={'YES' if r0.url else 'NO'}")

    return results
