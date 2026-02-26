"""
providers/ttn.py

TTN (tickets.ua) integration:
- Low-level HTTP helpers (ttn_get, ttn_post)
- Offer-to-FlightOption mapping
- Full scan entry point (run_ttn_scan)

TTN is the primary search provider for /search-business and alert scanning.
"""

import hashlib
import json as _json
import os
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import requests
from fastapi import HTTPException

from config import TTN_BASE_URL, get_config_str
from schemas.search import CabinClass, CabinSummary, FlightOption, SearchParams

try:
    from airlines import AIRLINE_NAMES, AIRLINE_BOOKING_URLS as AIRLINE_BOOKING_URL
    from airlines import AIRCRAFT_NAMES
except ImportError:
    AIRLINE_NAMES: Dict[str, str] = {}
    AIRLINE_BOOKING_URL: Dict[str, str] = {}
    AIRCRAFT_NAMES: Dict[str, str] = {}


# =====================================================================
# SECTION: AUTH HELPERS
# =====================================================================

def _get_ttn_api_key() -> Optional[str]:
    return os.getenv("TTN_API_KEY") or get_config_str("TTN_API_KEY", None)


def _get_ttn_auth_key() -> Optional[str]:
    return (os.getenv("TTN_AUTH_KEY") or "").strip() or None


def _ttn_headers() -> Dict[str, str]:
    return {
        "Accept": "application/json",
        "Content-Type": "application/json",
    }


# =====================================================================
# SECTION: LOW LEVEL HTTP HELPERS
# =====================================================================

def ttn_get(path: str, params: Optional[dict] = None) -> dict:
    api_key = _get_ttn_api_key()
    if not api_key:
        raise HTTPException(status_code=500, detail="TTN_API_KEY is not configured")

    url = f"{TTN_BASE_URL}{path}"
    merged_params = dict(params or {})
    merged_params.setdefault("key", api_key)

    print(f"[ttn] GET {path} params={merged_params}")

    res = requests.get(url, params=merged_params, headers=_ttn_headers(), timeout=30)

    if res.status_code >= 400:
        raise HTTPException(
            status_code=502,
            detail=f"TTN GET {path} failed: {res.status_code} {res.text}",
        )

    return res.json()


def ttn_post(path: str, payload: dict, params: Optional[dict] = None) -> dict:
    """Not used for /avia/search (GET-only), kept for future endpoints."""
    api_key = _get_ttn_api_key()
    if not api_key:
        raise HTTPException(status_code=500, detail="TTN_API_KEY is not configured")

    url = f"{TTN_BASE_URL}{path}"
    merged_params = dict(params or {})
    merged_params.setdefault("key", api_key)

    print(f"[ttn] POST {path} params={merged_params} payload_keys={list((payload or {}).keys())}")

    res = requests.post(url, params=merged_params, json=payload, headers=_ttn_headers(), timeout=30)

    if res.status_code >= 400:
        raise HTTPException(
            status_code=502,
            detail=f"TTN POST {path} failed: {res.status_code} {res.text}",
        )

    return res.json()


# =====================================================================
# SECTION: OFFER MAPPING
# =====================================================================

def map_ttn_offer_to_option(
    offer: dict,
    dep_date: date,
    ret_date: date,
    passengers: int,
    origin: str,
    destination: str,
    session_id: Optional[str] = None,
) -> FlightOption:
    """
    TTN recommendation -> FlightOption mapping (full card details).
    Builds outboundSegments / returnSegments compatible with the segment shape used by Base44.
    """
    if not isinstance(offer, dict):
        raise ValueError("TTN offer is not a dict")

    # ---- Price ----
    cur = offer.get("currency") or "EUR"

    amt = offer.get("amount")
    amt_val = None
    if isinstance(amt, dict) and cur:
        amt_val = amt.get(cur)
    else:
        amt_val = amt

    if amt_val is None:
        fare = offer.get("fare")
        taxes = offer.get("taxes")
        if fare is not None and taxes is not None:
            try:
                amt_val = float(fare) + float(taxes)
            except Exception:
                amt_val = None

    if amt_val is None:
        # Last resort: scan common price keys
        for k in ("total", "price", "totalPrice", "total_price", "cost"):
            v = offer.get(k)
            if v is not None:
                try:
                    amt_val = float(v)
                    break
                except Exception:
                    continue

    try:
        price = float(amt_val) if amt_val is not None else 0.0
    except Exception:
        price = 0.0

    # ---- Routes / segments ----
    routes = offer.get("routes") or []

    outbound_segments: List[Dict[str, Any]] = []
    return_segments: List[Dict[str, Any]] = []
    aircraft_codes: List[str] = []
    aircraft_names: List[str] = []
    all_airline_codes: List[str] = []
    outbound_total_minutes = 0
    return_total_minutes = 0

    def _parse_datetime(dt_str: Optional[str]) -> Optional[datetime]:
        if not dt_str:
            return None
        for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M", "%Y-%m-%d"):
            try:
                return datetime.strptime(dt_str, fmt)
            except Exception:
                continue
        return None

    def _process_route(route: dict, direction: str, idx: int) -> Tuple[List[Dict[str, Any]], int]:
        segs_out = []
        total_mins = 0

        # TTN routes can be dicts or lists
        segments = []
        if isinstance(route, dict):
            segments = route.get("segments") or route.get("flights") or []
            if not segments:
                # route itself might be a single segment
                segments = [route]
        elif isinstance(route, list):
            segments = route

        for seg_idx, seg in enumerate(segments):
            if not isinstance(seg, dict):
                continue

            dep_iata = seg.get("departure_airport") or seg.get("from") or seg.get("origin") or ""
            arr_iata = seg.get("arrival_airport") or seg.get("to") or seg.get("destination") or ""

            dep_at = seg.get("departure_date") or seg.get("departure_datetime") or seg.get("dep_date") or ""
            arr_at = seg.get("arrival_date") or seg.get("arrival_datetime") or seg.get("arr_date") or ""

            dep_dt = _parse_datetime(dep_at)
            arr_dt = _parse_datetime(arr_at)

            # Duration
            duration_mins = None
            raw_dur = seg.get("duration") or seg.get("flight_duration") or seg.get("duration_minutes")
            if raw_dur is not None:
                try:
                    duration_mins = int(float(raw_dur))
                except Exception:
                    pass
            if duration_mins is None and dep_dt and arr_dt:
                try:
                    duration_mins = max(0, int((arr_dt - dep_dt).total_seconds() // 60))
                except Exception:
                    pass

            if duration_mins:
                total_mins += duration_mins

            # Layover to next segment
            layover_mins: Optional[int] = None
            if seg_idx < len(segments) - 1:
                next_seg = segments[seg_idx + 1]
                next_dep_at = next_seg.get("departure_date") or next_seg.get("departure_datetime") or ""
                next_dep_dt = _parse_datetime(next_dep_at)
                if arr_dt and next_dep_dt:
                    try:
                        layover_mins = max(0, int((next_dep_dt - arr_dt).total_seconds() // 60))
                    except Exception:
                        pass
            if layover_mins:
                total_mins += layover_mins

            # Airline
            carrier_code = seg.get("airline_code") or seg.get("carrier") or seg.get("marketing_carrier") or ""
            if carrier_code:
                all_airline_codes.append(carrier_code)

            # Aircraft
            ac_code = seg.get("aircraft_code") or seg.get("aircraft") or seg.get("equipment") or ""
            if ac_code:
                aircraft_codes.append(ac_code)
                ac_name = AIRCRAFT_NAMES.get(ac_code)
                if ac_name:
                    aircraft_names.append(ac_name)

            # Cabin
            cabin_raw = seg.get("cabin") or seg.get("cabin_class") or seg.get("service_class") or ""
            cabin_norm = None
            if cabin_raw:
                cabin_upper = str(cabin_raw).upper()
                if cabin_upper in ("B", "BUSINESS", "J", "C"):
                    cabin_norm = "BUSINESS"
                elif cabin_upper in ("F", "FIRST", "P"):
                    cabin_norm = "FIRST"
                elif cabin_upper in ("W", "PREMIUM_ECONOMY", "PREMIUM"):
                    cabin_norm = "PREMIUM_ECONOMY"
                elif cabin_upper in ("E", "ECONOMY", "Y", "M", "S"):
                    cabin_norm = "ECONOMY"

            segs_out.append({
                "direction": direction,
                "flightNumber": seg.get("flight_number") or seg.get("number") or "",
                "marketingCarrier": carrier_code,
                "operatingCarrier": seg.get("operating_carrier") or carrier_code,
                "origin": dep_iata,
                "destination": arr_iata,
                "originAirport": seg.get("departure_airport_name") or "",
                "destinationAirport": seg.get("arrival_airport_name") or "",
                "departingAt": dep_at,
                "arrivingAt": arr_at,
                "aircraftCode": ac_code,
                "aircraftName": AIRCRAFT_NAMES.get(ac_code),
                "durationMinutes": duration_mins,
                "layoverMinutesToNext": layover_mins,
                "cabin": cabin_norm,
                "bookingCode": seg.get("booking_code") or seg.get("class"),
                "fareBrand": seg.get("fare_brand") or seg.get("fare_basis"),
            })

        return segs_out, total_mins

    for r_idx, route in enumerate(routes if isinstance(routes, list) else []):
        direction = "outbound" if r_idx == 0 else "return"
        segs, total_mins = _process_route(route, direction, r_idx)
        if direction == "outbound":
            outbound_segments = segs
            outbound_total_minutes = total_mins
        else:
            return_segments = segs
            return_total_minutes = total_mins

    # ---- Airline resolution ----
    # TTN sets a top-level airline_code or carrier on the recommendation
    top_carrier = offer.get("airline_code") or offer.get("carrier") or offer.get("validating_carrier") or ""
    if not top_carrier and all_airline_codes:
        top_carrier = all_airline_codes[0]

    airline_name = AIRLINE_NAMES.get(top_carrier, top_carrier or "Airline")
    booking_url = AIRLINE_BOOKING_URL.get(top_carrier) if isinstance(AIRLINE_BOOKING_URL, dict) else None

    # ---- Stops ----
    stops_outbound = max(0, len(outbound_segments) - 1) if outbound_segments else 0

    # ---- Stopovers ----
    stopover_codes: List[str] = []
    stopover_airports: List[str] = []
    if stops_outbound > 0 and outbound_segments:
        for seg in outbound_segments[:-1]:
            dest = seg.get("destination")
            dest_name = seg.get("destinationAirport")
            if dest:
                stopover_codes.append(dest)
            if dest_name:
                stopover_airports.append(dest_name)

    # ---- Origin / destination IATA ----
    origin_code = None
    destination_code = None
    origin_airport = None
    destination_airport = None

    if outbound_segments:
        first = outbound_segments[0]
        last = outbound_segments[-1]
        origin_code = first.get("origin") or origin
        destination_code = last.get("destination") or destination
        origin_airport = first.get("originAirport")
        destination_airport = last.get("destinationAirport")

    # ---- Cabin summary ----
    all_cabin_strs = [seg.get("cabin") for seg in outbound_segments + return_segments if seg.get("cabin")]
    cabin_vals = []
    for c in all_cabin_strs:
        if c and c.upper() in ("ECONOMY", "PREMIUM_ECONOMY", "BUSINESS", "FIRST"):
            try:
                cabin_vals.append(CabinClass(c.upper()))
            except Exception:
                pass

    if not cabin_vals:
        cabin_summary = CabinSummary.UNKNOWN
        cabin_highest = None
    elif len({v.value for v in cabin_vals}) == 1:
        cabin_summary = CabinSummary(cabin_vals[0].value)
        cabin_highest = cabin_vals[0]
    else:
        cabin_rank = {"ECONOMY": 1, "PREMIUM_ECONOMY": 2, "BUSINESS": 3, "FIRST": 4}
        cabin_highest = max(cabin_vals, key=lambda c: cabin_rank.get(c.value, 0))
        cabin_summary = CabinSummary.MIXED

    cabin_by_direction: Optional[Dict[str, Optional[CabinSummary]]] = None
    if outbound_segments or return_segments:
        ob_cabins_raw = [seg.get("cabin") for seg in outbound_segments if seg.get("cabin")]
        ret_cabins_raw = [seg.get("cabin") for seg in return_segments if seg.get("cabin")]

        def _to_summary(cabin_strs: List[str]) -> Optional[CabinSummary]:
            vals = []
            for c in cabin_strs:
                if c and c.upper() in ("ECONOMY", "PREMIUM_ECONOMY", "BUSINESS", "FIRST"):
                    try:
                        vals.append(CabinClass(c.upper()))
                    except Exception:
                        pass
            if not vals:
                return CabinSummary.UNKNOWN
            if len({v.value for v in vals}) == 1:
                return CabinSummary(vals[0].value)
            return CabinSummary.MIXED

        cabin_by_direction = {
            "outbound": _to_summary(ob_cabins_raw),
            "return": _to_summary(ret_cabins_raw),
        }

    # ---- Duration ----
    duration_minutes = outbound_total_minutes
    total_duration_minutes = outbound_total_minutes + return_total_minutes

    def iso8601_duration(minutes: int) -> str:
        mins = max(0, int(minutes or 0))
        h = mins // 60
        m = mins % 60
        if h and m:
            return f"PT{h}H{m}M"
        if h:
            return f"PT{h}H"
        return f"PT{m}M"

    iso_dur = iso8601_duration(total_duration_minutes)

    # ---- Stable ID ----
    id_src = _json.dumps({
        "session": session_id or "",
        "rec": offer.get("recommendation_id") or offer.get("id") or "",
        "dep": dep_date.isoformat(),
        "ret": ret_date.isoformat(),
        "carrier": top_carrier,
        "price": str(price),
    }, sort_keys=True)
    offer_id = hashlib.md5(id_src.encode()).hexdigest()

    # ---- Provider handoff ----
    # TTN requires session_id + recommendation_id for booking
    ttn_rec_id = offer.get("recommendation_id") or offer.get("id") or offer.get("rec_id")

    return FlightOption(
        id=offer_id,
        provider="ttn",
        providerSessionId=session_id,
        providerRecommendationId=str(ttn_rec_id) if ttn_rec_id else None,
        airline=airline_name,
        airlineCode=top_carrier or None,
        price=price,
        currency=cur,
        departureDate=dep_date.isoformat(),
        returnDate=ret_date.isoformat(),
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
        outboundSegments=outbound_segments or None,
        returnSegments=return_segments or None,
        aircraftCodes=aircraft_codes or None,
        aircraftNames=aircraft_names or None,
        bookingUrl=booking_url,
        url=booking_url,
    )


# =====================================================================
# SECTION: MAIN TTN SCAN
# =====================================================================

def run_ttn_scan(
    params: SearchParams,
    dep_override: Optional[date] = None,
    ret_override: Optional[date] = None,
) -> List[FlightOption]:
    print(f"[ttn] run_ttn_scan START origin={getattr(params,'origin',None)} dest={getattr(params,'destination',None)}")

    dep = dep_override or getattr(params, "earliestDeparture", None) or getattr(params, "departure_date", None)

    if not dep or not getattr(params, "origin", None) or not getattr(params, "destination", None):
        print("[ttn] missing required params (origin/destination/dep), skipping TTN scan")
        return []

    # TTN expects DD-MM-YYYY
    dep_str = dep.strftime("%d-%m-%Y") if hasattr(dep, "strftime") else str(dep)

    pax = int(getattr(params, "passengers", 1) or 1)

    # Map Flyyv cabin to TTN service_class
    cabin_raw = (getattr(params, "cabin", None) or "BUSINESS").upper()
    if cabin_raw in ("BUSINESS", "B"):
        service_class = "B"
    elif cabin_raw in ("ECONOMY", "E"):
        service_class = "E"
    else:
        service_class = "A"

    # Return date
    ret_date_obj = None
    if isinstance(ret_override, date):
        ret_date_obj = ret_override

    if ret_date_obj is None and hasattr(dep, "strftime"):
        try:
            nights = getattr(params, "minStayDays", None)
            if nights is None:
                nights = getattr(params, "nights", None)
            if nights is not None:
                nights = int(nights)
                if nights > 0:
                    ret_date_obj = dep + timedelta(days=nights)
        except Exception:
            ret_date_obj = None

    ret_str = ret_date_obj.strftime("%d-%m-%Y") if ret_date_obj else None

    qs: Dict[str, Any] = {
        "destinations[0][departure]": params.origin,
        "destinations[0][arrival]": params.destination,
        "destinations[0][date]": dep_str,
        "adt": pax,
        "service_class": service_class,
        "lang": "en",
    }

    if ret_str:
        qs.update({
            "destinations[1][departure]": params.destination,
            "destinations[1][arrival]": params.origin,
            "destinations[1][date]": ret_str,
        })

    res = None
    recs = None

    try:
        res = ttn_get("/avia/search.json", params=qs)

        if isinstance(res, dict) and "response" in res:
            resp = res.get("response", {}) or {}
            result = resp.get("result", {}) or {}
            session = resp.get("session", {}) or {}
            recs = resp.get("recommendations", None)

            rec_count = 0
            cheapest = None
            cheapest_currency = None

            if isinstance(recs, list):
                rec_count = len(recs)

                sample_printed = 0
                for r0 in recs:
                    if not isinstance(r0, dict):
                        continue

                    if sample_printed < 2:
                        keys = list(r0.keys())
                        print(f"[ttn] rec.sample_keys[{sample_printed}]={keys[:25]}")
                        print(
                            f"[ttn] rec.sample_prices[{sample_printed}] "
                            f"amount={r0.get('amount')} fare={r0.get('fare')} "
                            f"taxes={r0.get('taxes')} currency={r0.get('currency')}"
                        )
                        routes0 = r0.get("routes")
                        print(
                            f"[ttn] rec.sample_routes[{sample_printed}] "
                            f"type={type(routes0).__name__} preview={str(routes0)[:600]}"
                        )
                        cand_keys = []
                        for k in r0.keys():
                            kl = str(k).lower()
                            if any(s in kl for s in ["url", "link", "book", "pay", "payment", "order", "redirect", "checkout", "fat", "token"]):
                                cand_keys.append(k)
                        if cand_keys:
                            cand = {k: r0.get(k) for k in cand_keys}
                            print(f"[ttn] rec.checkout_candidates[{sample_printed}] keys={cand_keys} values_preview={str(cand)[:800]}")
                        else:
                            print(f"[ttn] rec.checkout_candidates[{sample_printed}] keys=[]")
                        sample_printed += 1

                    c_cur = r0.get("currency")
                    c_amt = r0.get("amount")
                    if isinstance(c_amt, dict) and c_cur:
                        c_val = c_amt.get(c_cur)
                    else:
                        c_val = c_amt

                    if c_val is None:
                        c_fare = r0.get("fare")
                        c_taxes = r0.get("taxes")
                        if c_fare is not None and c_taxes is not None:
                            try:
                                c_val = float(c_fare) + float(c_taxes)
                            except Exception:
                                c_val = None

                    if c_val is None:
                        continue
                    try:
                        val = float(c_val)
                    except Exception:
                        continue

                    if cheapest is None or val < cheapest:
                        cheapest = val
                        cheapest_currency = c_cur

            elif isinstance(recs, dict):
                rec_count = len(recs)

            print(f"[ttn] result.code={result.get('code')} desc={result.get('description')}")
            print(
                f"[ttn] session.id={session.get('id')} recs={rec_count} "
                f"cheapest={cheapest} {cheapest_currency} service_class={service_class} dep={dep_str}"
            )
        else:
            print("[ttn] unexpected response type/shape:", type(res), "sample:", str(res)[:800])

    except Exception as e:
        print(f"[ttn] avia/search failed: {e}")

    print("[ttn] run_ttn_scan END")

    # ---- Map and return ----
    try:
        if not isinstance(res, dict):
            print(f"[ttn] mapping_skip res_type={type(res).__name__}")
            return []

        resp = (res.get("response") or {}) if isinstance(res.get("response"), dict) else {}
        recs = resp.get("recommendations")
        session_id = (resp.get("session") or {}).get("id") if isinstance(resp.get("session"), dict) else None

        if not isinstance(recs, list) or not recs:
            print("[ttn] mapping_skip recs_empty_or_invalid")
            return []

        # Normalize dep date
        if isinstance(dep, date):
            dep_date_obj = dep
        else:
            try:
                dep_date_obj = datetime.fromisoformat(str(dep)).date()
            except Exception:
                dep_date_obj = date.today()

        ret_date_obj_final = None
        if isinstance(ret_override, date):
            ret_date_obj_final = ret_override
        else:
            try:
                nights = getattr(params, "minStayDays", None)
                if nights is None:
                    nights = getattr(params, "nights", None)
                if nights is not None:
                    nights = int(nights)
                    if nights > 0:
                        ret_date_obj_final = dep_date_obj + timedelta(days=nights)
            except Exception:
                ret_date_obj_final = None

        if ret_date_obj_final is None:
            ret_date_obj_final = dep_date_obj

        mapped: List[FlightOption] = []
        attempted = 0

        max_per_pair = int(os.getenv("MAX_OFFERS_PER_PAIR", "20"))
        for r0 in recs:
            if len(mapped) >= max_per_pair:
                break

            attempted += 1

            if not isinstance(r0, dict):
                continue

            try:
                opt = map_ttn_offer_to_option(
                    r0,
                    dep_date=dep_date_obj,
                    ret_date=ret_date_obj_final,
                    passengers=pax,
                    origin=str(params.origin),
                    destination=str(params.destination),
                    session_id=session_id,
                )
                mapped.append(opt)
            except Exception as e:
                print(f"[ttn] map_failed idx={attempted-1} err={type(e).__name__}: {e}")
                continue

        if not mapped:
            print(f"[ttn] mapped=0 attempted={attempted} session_id={session_id}")
            return []

        o0 = mapped[0]
        print(
            f"[ttn] mapped={len(mapped)} attempted={attempted} session_id={session_id} "
            f"first_id={getattr(o0,'id',None)} first_airline={getattr(o0,'airline',None)} "
            f"first_price={getattr(o0,'price',None)} {getattr(o0,'currency',None)}"
        )

        return mapped

    except Exception as e:
        print(f"[ttn] mapping_block_failed err={type(e).__name__}: {e}")
        return []
