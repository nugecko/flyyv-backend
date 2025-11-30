import os
from datetime import date, datetime, timedelta
from enum import Enum
from typing import List, Optional, Tuple, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from uuid import uuid4
from collections import defaultdict
import smtplib
from email.message import EmailMessage

import requests
from fastapi import FastAPI, Header, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from db import engine, Base, SessionLocal
import models  # noqa: F401
from models import AdminConfig, AppUser

# =======================================
# SECTION: AIRLINES IMPORTS
# =======================================

try:
    from airlines import AIRLINE_NAMES  # type: ignore
except ImportError:
    AIRLINE_NAMES: Dict[str, str] = {}

try:
    from airlines import AIRLINE_BOOKING_URL  # type: ignore
except ImportError:
    try:
        from airlines import AIRLINE_BOOKING_URLS as AIRLINE_BOOKING_URL  # type: ignore
    except ImportError:
        AIRLINE_BOOKING_URL: Dict[str, str] = {}

# ===== END SECTION: AIRLINES IMPORTS =====


# =======================================
# SECTION: ADMIN CONFIG HELPERS
# =======================================

def _get_config_row(db: Session, key: str) -> Optional[AdminConfig]:
    return db.query(AdminConfig).filter(AdminConfig.key == key).first()


def get_config_str(key: str, default_value: Optional[str] = None) -> Optional[str]:
    """
    Read a config value from admin_config as string.
    If the key is missing or value is null, return default_value.
    """
    db = SessionLocal()
    try:
        row = _get_config_row(db, key)
        if not row or row.value is None:
            return default_value
        return str(row.value)
    finally:
        db.close()


def get_config_int(key: str, default_value: int) -> int:
    """
    Read a config value from admin_config and cast to int.
    Falls back to default_value if missing or invalid.
    """
    raw = get_config_str(key, None)
    if raw is None:
        return default_value
    try:
        return int(raw)
    except ValueError:
        return default_value

# ===== END SECTION: ADMIN CONFIG HELPERS =====


# =======================================
# SECTION: Pydantic MODELS
# =======================================

class SearchParams(BaseModel):
    origin: str
    destination: str
    earliestDeparture: date
    latestDeparture: date
    minStayDays: int
    maxStayDays: int
    maxPrice: Optional[float] = None
    cabin: str = "BUSINESS"
    passengers: int = 1
    stopsFilter: Optional[List[int]] = None

    maxOffersPerPair: int = 50
    maxOffersTotal: int = 5000
    maxDatePairs: int = 45

    fullCoverage: bool = True


class FlightOption(BaseModel):
    id: str
    airline: str
    airlineCode: Optional[str] = None
    price: float
    currency: str
    departureDate: str
    returnDate: str
    stops: int

    durationMinutes: int
    totalDurationMinutes: Optional[int] = None
    duration: Optional[str] = None

    origin: Optional[str] = None
    destination: Optional[str] = None
    originAirport: Optional[str] = None
    destinationAirport: Optional[str] = None

    stopoverCodes: Optional[List[str]] = None
    stopoverAirports: Optional[List[str]] = None

    segments: Optional[List[Dict[str, Any]]] = None

    aircraftCodes: Optional[List[str]] = None
    aircraftNames: Optional[List[str]] = None

    bookingUrl: Optional[str] = None
    url: Optional[str] = None


class CreditUpdateRequest(BaseModel):
    userId: str
    amount: Optional[int] = None
    delta: Optional[int] = None
    creditAmount: Optional[int] = None
    value: Optional[int] = None
    reason: Optional[str] = None


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class SearchJob(BaseModel):
    id: str
    status: JobStatus
    created_at: datetime
    updated_at: datetime
    params: SearchParams
    total_pairs: int = 0
    processed_pairs: int = 0
    error: Optional[str] = None


class SearchStatusResponse(BaseModel):
    jobId: str
    status: JobStatus
    processedPairs: int
    totalPairs: int
    progress: float
    error: Optional[str] = None
    previewCount: int = 0
    previewOptions: List[FlightOption] = Field(default_factory=list)


class SearchResultsResponse(BaseModel):
    jobId: str
    status: JobStatus
    totalResults: int
    offset: int
    limit: int
    options: List[FlightOption]


class UserSyncPayload(BaseModel):
    external_id: str
    email: str
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    country: Optional[str] = None
    marketing_consent: Optional[bool] = None
    source: Optional[str] = None


class ProfileUser(BaseModel):
    id: str
    name: Optional[str] = None
    email: Optional[str] = None
    plan: Optional[str] = None


class SubscriptionInfo(BaseModel):
    plan: str
    billingCycle: Optional[str] = None
    renewalDate: Optional[str] = None
    monthlyCredits: int = 0


class WalletInfo(BaseModel):
    balance: int
    currency: str = "credits"


class ProfileResponse(BaseModel):
    user: ProfileUser
    subscription: SubscriptionInfo
    wallet: WalletInfo


class PublicConfig(BaseModel):
    maxDepartureWindowDays: int
    maxStayNights: int
    minStayNights: int
    maxPassengers: int

# ===== END SECTION: Pydantic MODELS =====


# =======================================
# SECTION: FastAPI APP AND CORS
# =======================================

app = FastAPI()


@app.on_event("startup")
def on_startup():
    Base.metadata.create_all(bind=engine)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===== END SECTION: FastAPI APP AND CORS =====


# =======================================
# SECTION: ENV, DUFFEL AND EMAIL CONFIG
# =======================================

ADMIN_API_TOKEN = os.getenv("ADMIN_API_TOKEN")

DUFFEL_ACCESS_TOKEN = os.getenv("DUFFEL_ACCESS_TOKEN")
DUFFEL_API_BASE = "https://api.duffel.com"
DUFFEL_VERSION = "v2"

if not DUFFEL_ACCESS_TOKEN:
    print("WARNING: DUFFEL_ACCESS_TOKEN is not set, searches will fail")

MAX_OFFERS_PER_PAIR_HARD = 40
MAX_OFFERS_TOTAL_HARD = 500
MAX_DATE_PAIRS_HARD = 20

SYNC_PAIR_THRESHOLD = 10
PARALLEL_WORKERS = 2

# Email alert configuration for SMTP2Go
SMTP_HOST = os.getenv("SMTP_HOST", "mail-eu.smtp2go.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", "2525"))
SMTP_USERNAME = os.getenv("SMTP_USERNAME")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD")
ALERT_FROM_EMAIL = os.getenv("ALERT_FROM_EMAIL", "price-alert@flyyv.com")
ALERT_TO_EMAIL = os.getenv("ALERT_TO_EMAIL")

# Frontend base URL for deep links in alert emails
FRONTEND_BASE_URL = os.getenv("FRONTEND_BASE_URL", "https://app.flyyv.com")

# Simple single price watch config for testing
WATCH_ORIGIN = os.getenv("WATCH_ORIGIN", "LON")
WATCH_DESTINATION = os.getenv("WATCH_DESTINATION", "TLV")
WATCH_START_DATE = os.getenv("WATCH_START_DATE")  # YYYY-MM-DD
WATCH_END_DATE = os.getenv("WATCH_END_DATE")      # YYYY-MM-DD
WATCH_STAY_NIGHTS = int(os.getenv("WATCH_STAY_NIGHTS", "7"))
WATCH_MAX_PRICE = float(os.getenv("WATCH_MAX_PRICE", "720"))

# Global toggle for alerts
ALERTS_ENABLED = os.getenv("ALERTS_ENABLED", "true").lower() == "true"

# ===== END SECTION: ENV, DUFFEL AND EMAIL CONFIG =====


# =======================================
# SECTION: IN MEMORY STORES
# =======================================

USER_WALLETS: Dict[str, int] = {}
JOBS: Dict[str, SearchJob] = {}
JOB_RESULTS: Dict[str, List[FlightOption]] = {}

# ===== END SECTION: IN MEMORY STORES =====


# =======================================
# SECTION: DUFFEL HELPERS
# =======================================

def duffel_headers() -> dict:
    return {
        "Authorization": f"Bearer {DUFFEL_ACCESS_TOKEN}",
        "Accept": "application/json",
        "Content-Type": "application/json",
        "Duffel-Version": DUFFEL_VERSION,
    }


def generate_date_pairs(params: SearchParams, max_pairs: int = 60) -> List[Tuple[date, date]]:
    pairs: List[Tuple[date, date]] = []

    min_stay = max(1, params.minStayDays)
    max_stay = max(min_stay, params.maxStayDays)

    if params.earliestDeparture == params.latestDeparture and min_stay == max_stay:
        dep = params.earliestDeparture
        ret = dep + timedelta(days=min_stay)
        pairs.append((dep, ret))
        return pairs[:max_pairs]

    stays = list(range(min_stay, max_stay + 1))
    current = params.earliestDeparture

    while current <= params.latestDeparture and len(pairs) < max_pairs:
        for stay in stays:
            ret = current + timedelta(days=stay)
            if ret <= params.latestDeparture:
                pairs.append((current, ret))
                if len(pairs) >= max_pairs:
                    break
        current += timedelta(days=1)

    return pairs


def duffel_create_offer_request(
    slices: List[dict],
    passengers: List[dict],
    cabin_class: str,
) -> dict:
    if not DUFFEL_ACCESS_TOKEN:
        raise HTTPException(status_code=500, detail="Duffel not configured")

    url = f"{DUFFEL_API_BASE}/air/offer_requests"
    payload = {
        "data": {
            "slices": slices,
            "passengers": passengers,
            "cabin_class": cabin_class.lower(),
        }
    }

    resp = requests.post(url, json=payload, headers=duffel_headers(), timeout=10)
    if resp.status_code >= 400:
        print("Duffel offer_requests error:", resp.status_code, resp.text)
        raise HTTPException(status_code=502, detail="Duffel API error")

    body = resp.json()
    return body.get("data", {})


def duffel_list_offers(offer_request_id: str, limit: int = 300) -> List[dict]:
    url = f"{DUFFEL_API_BASE}/air/offers"
    params = {
        "offer_request_id": offer_request_id,
        "limit": min(limit, 300),
        "sort": "total_amount",
    }

    resp = requests.get(url, params=params, headers=duffel_headers(), timeout=10)
    if resp.status_code >= 400:
        print("Duffel offers error:", resp.status_code, resp.text)
        raise HTTPException(status_code=502, detail="Duffel API error")

    body = resp.json()
    data = body.get("data", [])
    return list(data)[:limit]


def build_iso_duration(minutes: int) -> str:
    if minutes <= 0:
        return "PT0M"
    hours = minutes // 60
    mins = minutes % 60
    if hours and mins:
        return f"PT{hours}H{mins}M"
    if hours:
        return f"PT{hours}H"
    return f"PT{mins}M"


def map_duffel_offer_to_option(
    offer: dict,
    dep: date,
    ret: date,
) -> FlightOption:
    price = float(offer.get("total_amount", 0))
    currency = offer.get("total_currency", "GBP")

    owner = offer.get("owner", {}) or {}
    airline_code = owner.get("iata_code")
    airline_name = AIRLINE_NAMES.get(
        airline_code,
        owner.get("name", airline_code or "Airline"),
    )
    booking_url = AIRLINE_BOOKING_URL.get(airline_code) if isinstance(AIRLINE_BOOKING_URL, dict) else None

    slices = offer.get("slices", []) or []
    outbound_segments: List[dict] = []
    return_segments: List[dict] = []

    if len(slices) >= 1:
        outbound_segments = slices[0].get("segments", []) or []
    if len(slices) >= 2:
        return_segments = slices[1].get("segments", []) or []

    stops_outbound = max(0, len(outbound_segments) - 1)

    duration_minutes = 0
    origin_code = None
    destination_code = None
    origin_airport = None
    destination_airport = None

    if outbound_segments:
        first_segment = outbound_segments[0]
        last_segment = outbound_segments[-1]

        origin_obj = first_segment.get("origin", {}) or {}
        dest_obj = last_segment.get("destination", {}) or {}

        origin_code = origin_obj.get("iata_code")
        destination_code = dest_obj.get("iata_code")
        origin_airport = origin_obj.get("name")
        destination_airport = dest_obj.get("name")

        dep_at = first_segment.get("departing_at")
        arr_at = last_segment.get("arriving_at")

        try:
            dep_dt = datetime.fromisoformat(dep_at.replace("Z", "+00:00"))
            arr_dt = datetime.fromisoformat(arr_at.replace("Z", "+00:00"))
            duration_minutes = int((arr_dt - dep_dt).total_seconds() // 60)
        except Exception:
            duration_minutes = 0

    iso_duration = build_iso_duration(duration_minutes)

    stopover_codes: List[str] = []
    stopover_airports: List[str] = []
    if len(outbound_segments) > 1:
        for seg in outbound_segments[:-1]:
            dest_obj = seg.get("destination", {}) or {}
            code = dest_obj.get("iata_code")
            name = dest_obj.get("name")
            if code:
                stopover_codes.append(code)
            if name:
                stopover_airports.append(name)

    segments_info: List[Dict[str, Any]] = []
    aircraft_codes: List[str] = []
    aircraft_names: List[str] = []

    for direction, seg_list in (("outbound", outbound_segments), ("return", return_segments)):
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

            layover_minutes_to_next: Optional[int] = None
            if idx < len(seg_list) - 1:
                this_arr = seg.get("arriving_at")
                next_dep = seg_list[idx + 1].get("departing_at")
                try:
                    this_arr_dt = datetime.fromisoformat(this_arr.replace("Z", "+00:00"))
                    next_dep_dt = datetime.fromisoformat(next_dep.replace("Z", "+00:00"))
                    layover_minutes_to_next = int((next_dep_dt - this_arr_dt).total_seconds() // 60)
                except Exception:
                    layover_minutes_to_next = None

            segments_info.append(
                {
                    "direction": direction,
                    "origin": o.get("iata_code"),
                    "originAirport": o.get("name"),
                    "destination": d.get("iata_code"),
                    "destinationAirport": d.get("name"),
                    "departingAt": seg.get("departing_at"),
                    "arrivingAt": seg.get("arriving_at"),
                    "aircraftCode": aircraft_code,
                    "aircraftName": aircraft_name,
                    "layoverMinutesToNext": layover_minutes_to_next,
                }
            )

    return FlightOption(
        id=offer.get("id", ""),
        airline=airline_name,
        airlineCode=airline_code or None,
        price=price,
        currency=currency,
        departureDate=dep.isoformat(),
        returnDate=ret.isoformat(),
        stops=stops_outbound,
        durationMinutes=duration_minutes,
        totalDurationMinutes=duration_minutes,
        duration=iso_duration,
        origin=origin_code,
        destination=destination_code,
        originAirport=origin_airport,
        destinationAirport=destination_airport,
        stopoverCodes=stopover_codes or None,
        stopoverAirports=stopover_airports or None,
        segments=segments_info or None,
        aircraftCodes=aircraft_codes or None,
        aircraftNames=aircraft_names or None,
        bookingUrl=booking_url,
        url=booking_url,
    )

# ===== END SECTION: DUFFEL HELPERS =====


# =======================================
# SECTION: FILTERING AND BALANCING
# =======================================

def apply_filters(options: List[FlightOption], params: SearchParams) -> List[FlightOption]:
    filtered = list(options)

    if params.maxPrice is not None and params.maxPrice > 0:
        filtered = [o for o in filtered if o.price <= params.maxPrice]

    if params.stopsFilter:
        allowed = set(params.stopsFilter)
        if 3 in allowed:
            filtered = [o for o in filtered if (o.stops in allowed or o.stops >= 3)]
        else:
            filtered = [o for o in filtered if o.stops in allowed]

    filtered.sort(key=lambda x: x.price)
    return filtered


def balance_airlines(options: List[FlightOption], max_per_airline: int = 50) -> List[FlightOption]:
    buckets: Dict[str, List[FlightOption]] = {}
    for opt in options:
        key = opt.airlineCode or opt.airline
        buckets.setdefault(key, []).append(opt)

    trimmed: List[FlightOption] = []
    for key, bucket in buckets.items():
        trimmed.extend(bucket[:max_per_airline])

    trimmed.sort(key=lambda x: x.price)
    return trimmed

# ===== END SECTION: FILTERING AND BALANCING =====


# =======================================
# SECTION: SHARED SEARCH HELPERS
# =======================================

def effective_caps(params: SearchParams) -> Tuple[int, int, int]:
    max_pairs = max(1, min(params.maxDatePairs, MAX_DATE_PAIRS_HARD))

    requested_per_pair = max(1, params.maxOffersPerPair)
    requested_total = max(1, params.maxOffersTotal)

    config_max_offers_pair = get_config_int("MAX_OFFERS_PER_PAIR", 50)
    config_max_offers_total = get_config_int("MAX_OFFERS_TOTAL", 5000)

    max_offers_pair = max(
        1,
        min(requested_per_pair, config_max_offers_pair, MAX_OFFERS_PER_PAIR_HARD),
    )
    max_offers_total = max(
        1,
        min(requested_total, config_max_offers_total, MAX_OFFERS_TOTAL_HARD),
    )

    return max_pairs, max_offers_pair, max_offers_total


def estimate_date_pairs(params: SearchParams) -> int:
    max_pairs, _, _ = effective_caps(params)
    pairs = generate_date_pairs(params, max_pairs=max_pairs)
    return len(pairs)


def run_duffel_scan(params: SearchParams) -> List[FlightOption]:
    max_pairs, max_offers_pair, max_offers_total = effective_caps(params)
    date_pairs = generate_date_pairs(params, max_pairs=max_pairs)
    if not date_pairs:
        return []

    collected_offers: List[Tuple[dict, date, date]] = []
    total_count = 0

    for dep, ret in date_pairs:
        if total_count >= max_offers_total:
            break

        slices = [
            {"origin": params.origin, "destination": params.destination, "departure_date": dep.isoformat()},
            {"origin": params.destination, "destination": params.origin, "departure_date": ret.isoformat()},
        ]
        pax = [{"type": "adult"} for _ in range(params.passengers)]

        try:
            offer_request = duffel_create_offer_request(slices, pax, params.cabin)
            offer_request_id = offer_request.get("id")
            if not offer_request_id:
                continue

            per_pair_limit = min(max_offers_pair, max_offers_total - total_count)
            offers_json = duffel_list_offers(offer_request_id, limit=per_pair_limit)
        except HTTPException as e:
            print("Duffel error for", dep, "to", ret, ":", e.detail)
            continue
        except Exception as e:
            print("Unexpected Duffel error for", dep, "to", ret, ":", e)
            continue

        for offer in offers_json:
            collected_offers.append((offer, dep, ret))
            total_count += 1
            if total_count >= max_offers_total:
                break

    mapped: List[FlightOption] = [
        map_duffel_offer_to_option(offer, dep, ret)
        for offer, dep, ret in collected_offers
    ]

    filtered = apply_filters(mapped, params)
    balanced = balance_airlines(filtered, max_per_airline=50)
    return balanced

# ===== END SECTION: SHARED SEARCH HELPERS =====


# =======================================
# SECTION: ASYNC JOB RUNNER
# =======================================

def process_date_pair_offers(
    params: SearchParams,
    dep: date,
    ret: date,
    max_offers_pair: int,
) -> List[FlightOption]:
    slices = [
        {
            "origin": params.origin,
            "destination": params.destination,
            "departure_date": dep.isoformat(),
        },
        {
            "origin": params.destination,
            "destination": params.origin,
            "departure_date": ret.isoformat(),
        },
    ]
    pax = [{"type": "adult"} for _ in range(params.passengers)]

    try:
        offer_request = duffel_create_offer_request(slices, pax, params.cabin)
        offer_request_id = offer_request.get("id")
        if not offer_request_id:
            print(f"[PAIR {dep} -> {ret}] No offer_request id")
            return []

        offers_json = duffel_list_offers(offer_request_id, limit=max_offers_pair)
    except HTTPException as e:
        print(f"[PAIR {dep} -> {ret}] Duffel HTTPException: {e.detail}")
        return []
    except Exception as e:
        print(f"[PAIR {dep} -> {ret}] Unexpected Duffel error: {e}")
        return []

    batch_mapped: List[FlightOption] = [
        map_duffel_offer_to_option(offer, dep, ret) for offer in offers_json
    ]
    return batch_mapped


def run_search_job(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        print(f"[JOB {job_id}] Job not found in memory")
        return

    job.status = JobStatus.RUNNING
    job.updated_at = datetime.utcnow()
    JOBS[job_id] = job
    print(f"[JOB {job_id}] Starting async search")

    if job_id not in JOB_RESULTS:
        JOB_RESULTS[job_id] = []

    try:
        max_pairs, max_offers_pair, max_offers_total = effective_caps(job.params)
        date_pairs = generate_date_pairs(job.params, max_pairs=max_pairs)
        total_pairs = len(date_pairs)
        job.total_pairs = total_pairs
        job.processed_pairs = 0
        job.updated_at = datetime.utcnow()
        JOBS[job_id] = job
        print(
            f"[JOB {job_id}] total_pairs={total_pairs}, "
            f"max_offers_pair={max_offers_pair}, max_offers_total={max_offers_total}"
        )

        total_count = 0

        if total_pairs == 0:
            job.status = JobStatus.COMPLETED
            job.updated_at = datetime.utcnow()
            JOBS
