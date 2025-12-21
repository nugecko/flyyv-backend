# =====================================================================
# SECTION START: IMPORTS
# =====================================================================

import os
from datetime import date, datetime, timedelta
from enum import Enum
from typing import List, Optional, Tuple, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from uuid import uuid4
from collections import defaultdict, Counter
import smtplib
from email.message import EmailMessage

import requests
from fastapi import FastAPI, Header, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from db import engine, Base, SessionLocal
import models  # noqa: F401
from models import AdminConfig, AppUser, Alert, AlertRun
from alerts_email import (
    send_alert_email_for_alert,
    send_smart_alert_email,
    send_alert_confirmation_email,
)

# =====================================================================
# SECTION END: IMPORTS
# =====================================================================


# =====================================================================
# SECTION START: ALERT TOGGLES
# =====================================================================

def master_alerts_enabled() -> bool:
    """
    Hard master switch controlled by environment variable ALERTS_ENABLED.
    If this is set to 'false' (case insensitive), alerts are completely disabled.
    """
    value = os.getenv("ALERTS_ENABLED", "true")
    return value.lower() == "true"


def alerts_globally_enabled(db: Session) -> bool:
    """
    Global switch stored in admin_config with key = 'GLOBAL_ALERTS'.
    If the row does not exist, default to True.
    """
    config = db.query(AdminConfig).filter(AdminConfig.key == "GLOBAL_ALERTS").first()
    if not config:
        return True
    if not hasattr(config, "alerts_enabled"):
        return True
    return bool(config.alerts_enabled)


def user_allows_alerts(user: AppUser) -> bool:
    """
    Per user toggle, defaults to True if the column is missing.
    """
    if not hasattr(user, "email_alerts_enabled"):
        return True
    return bool(user.email_alerts_enabled)


def should_send_alert(db: Session, user: AppUser) -> bool:
    """
    Combined logic:
    1. Environment master toggle must be ON
    2. Global admin_config toggle must be ON
    3. User toggle must be ON
    """
    if not master_alerts_enabled():
        return False
    if not alerts_globally_enabled(db):
        return False
    if not user_allows_alerts(user):
        return False
    return True

# =====================================================================
# SECTION END: ALERT TOGGLES
# =====================================================================


# =====================================================================
# SECTION START: AIRLINES IMPORTS
# =====================================================================

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

# =====================================================================
# SECTION END: AIRLINES IMPORTS
# =====================================================================


# =====================================================================
# SECTION START: ADMIN CONFIG HELPERS
# =====================================================================

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


def get_config_bool(key: str, default_value: bool) -> bool:
    """
    Read a config value from admin_config and cast to bool.
    Accepts values like 1, 0, true, false, yes, no, on, off.
    Falls back to default_value if missing or invalid.
    """
    raw = get_config_str(key, None)
    if raw is None:
        return default_value

    s = str(raw).strip().lower()
    if s in ("1", "true", "yes", "y", "on"):
        return True
    if s in ("0", "false", "no", "n", "off"):
        return False
    return default_value

# =====================================================================
# SECTION END: ADMIN CONFIG HELPERS
# =====================================================================


# =====================================================================
# SECTION START: Pydantic MODELS
# =====================================================================

class AlertCreate(BaseModel):
    email: str
    origin: str
    destination: str
    cabin: str

    search_mode: Optional[str] = "flexible"  # "flexible" or "fixed"
    departure_start: Optional[date] = None
    departure_end: Optional[date] = None
    return_start: Optional[date] = None
    return_end: Optional[date] = None

    alert_type: Optional[str] = None
    max_price: Optional[int] = None

    mode: Optional[str] = "single"  # "single" or "smart"
    passengers: Optional[int] = 1


class AlertOut(BaseModel):
    id: str
    email: str
    origin: str
    destination: str
    cabin: str
    search_mode: str

    departure_start: Optional[date] = None
    departure_end: Optional[date] = None
    return_start: Optional[date] = None
    return_end: Optional[date] = None

    alert_type: Optional[str] = None
    max_price: Optional[int] = None
    mode: str

    passengers: int
    times_sent: int
    is_active: bool

    last_price: Optional[int] = None
    best_price: Optional[int] = None

    last_run_at: Optional[datetime] = None
    last_notified_at: Optional[datetime] = None
    last_notified_price: Optional[int] = None

    created_at: datetime
    updated_at: datetime

class CreditUpdateRequest(BaseModel):
    userId: str

    # Accept any of these keys from the admin UI/tooling
    delta: Optional[int] = None
    amount: Optional[int] = None
    creditAmount: Optional[int] = None
    value: Optional[int] = None

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

    maxOffersPerPair: int = 300
    maxOffersTotal: int = 10000
    maxDatePairs: int = 60


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class CabinClass(str, Enum):
    ECONOMY = "ECONOMY"
    PREMIUM_ECONOMY = "PREMIUM_ECONOMY"
    BUSINESS = "BUSINESS"
    FIRST = "FIRST"

class CabinSummary(str, Enum):
    ECONOMY = "ECONOMY"
    PREMIUM_ECONOMY = "PREMIUM_ECONOMY"
    BUSINESS = "BUSINESS"
    FIRST = "FIRST"
    MIXED = "MIXED"
    UNKNOWN = "UNKNOWN"


class SearchJob(BaseModel):
    id: str
    status: JobStatus
    created_at: datetime
    updated_at: datetime
    params: SearchParams
    total_pairs: int = 0
    processed_pairs: int = 0
    error: Optional[str] = None


class FlightOption(BaseModel):
    id: str
    airline: str
    airlineCode: Optional[str] = None
    price: float
    currency: str
    departureDate: str
    returnDate: str
    stops: int

    cabinSummary: Optional[CabinSummary] = None
    cabinHighest: Optional[CabinClass] = None
    cabinByDirection: Optional[Dict[str, Optional[CabinSummary]]] = None

    durationMinutes: int
    totalDurationMinutes: Optional[int] = None
    duration: Optional[str] = None

    origin: Optional[str] = None
    destination: Optional[str] = None
    originAirport: Optional[str] = None
    destinationAirport: Optional[str] = None

    stopoverCodes: Optional[List[str]] = None
    stopoverAirports: Optional[List[str]] = None

    outboundSegments: Optional[List[Dict[str, Any]]] = None
    returnSegments: Optional[List[Dict[str, Any]]] = None

    aircraftCodes: Optional[List[str]] = None
    aircraftNames: Optional[List[str]] = None

    bookingUrl: Optional[str] = None
    url: Optional[str] = None


class SearchStartResponse(BaseModel):
    status: JobStatus
    mode: str
    jobId: str


class SearchStatusResponse(BaseModel):
    jobId: str
    status: JobStatus
    processedPairs: int
    totalPairs: int
    progress: float
    error: Optional[str]
    previewCount: int
    previewOptions: List[FlightOption]


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


class PublicUser(BaseModel):
    id: str
    external_id: Optional[str] = None
    email: str
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    country: Optional[str] = None
    marketing_consent: Optional[bool] = None
    source: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


class AdminConfigResponse(BaseModel):
    key: str
    value: str
    updated_at: Optional[datetime] = None


class AdminConfigUpdatePayload(BaseModel):
    value: str


class EarlyAccessCreatePayload(BaseModel):
    email: str


class EarlyAccessResponse(BaseModel):
    email: str
    created_at: datetime


class AlertBase(BaseModel):
    email: str
    origin: str
    destination: str
    cabin: str
    search_mode: Optional[str] = "flexible"
    preferred_days: Optional[List[int]] = None
    max_price: Optional[int] = None
    min_days: Optional[int] = None
    max_days: Optional[int] = None
    notes: Optional[str] = None
    currency: Optional[str] = "GBP"


class AlertCreatePayload(AlertBase):
    pass


class AlertUpdatePayload(BaseModel):
    is_active: Optional[bool] = None
    preferred_days: Optional[List[int]] = None
    max_price: Optional[int] = None
    min_days: Optional[int] = None
    max_days: Optional[int] = None
    notes: Optional[str] = None


class AlertResponse(AlertBase):
    id: str
    is_active: bool
    created_at: datetime
    updated_at: datetime

    class Config:
        orm_mode = True


class AlertRunResponse(BaseModel):
    id: str
    alert_id: str
    ran_at: datetime
    matches_count: int
    min_price_found: Optional[int] = None
    max_price_found: Optional[int] = None
    currency: Optional[str] = "GBP"

    class Config:
        orm_mode = True


class ProfileUser(BaseModel):
    id: str
    email: str
    credits: int


class SubscriptionInfo(BaseModel):
    plan: str
    status: str
    renews_on: Optional[str] = None


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


class AlertWithStatsResponse(AlertResponse):
    # Computed from AlertRun history (min price_found)
    best_price: Optional[int] = None

    # Timestamps for UI status fields
    last_run_at: Optional[datetime] = None
    last_notified_at: Optional[datetime] = None
    last_notified_price: Optional[int] = None

    created_at: datetime
    updated_at: datetime

    class Config:
        orm_mode = True

# =====================================================================
# SECTION END: Pydantic MODELS
# =====================================================================

# =====================================================================
# SECTION START: FastAPI APP AND CORS
# =====================================================================

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

from early_access import router as early_access_router
app.include_router(early_access_router)

# =====================================================================
# SECTION END: FastAPI APP AND CORS
# =====================================================================

# =====================================================================
# SECTION START: ENV, DUFFEL AND EMAIL CONFIG
# =====================================================================

ADMIN_API_TOKEN = os.getenv("ADMIN_API_TOKEN")

DUFFEL_ACCESS_TOKEN = os.getenv("DUFFEL_ACCESS_TOKEN")
DUFFEL_API_BASE = "https://api.duffel.com"
DUFFEL_VERSION = "v2"

if not DUFFEL_ACCESS_TOKEN:
    print("WARNING: DUFFEL_ACCESS_TOKEN is not set, searches will fail")

MAX_OFFERS_PER_PAIR_HARD = 300
MAX_OFFERS_TOTAL_HARD = 4000
MAX_DATE_PAIRS_HARD = 60

SYNC_PAIR_THRESHOLD = 10
PARALLEL_WORKERS = 6

SMTP_HOST = os.getenv("SMTP_HOST", "mail-eu.smtp2go.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", "2525"))
SMTP_USERNAME = os.getenv("SMTP_USERNAME")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD")
ALERT_FROM_EMAIL = os.getenv("ALERT_FROM_EMAIL", "price-alert@flyyv.com")
ALERT_TO_EMAIL = os.getenv("ALERT_TO_EMAIL")

FRONTEND_BASE_URL = os.getenv("FRONTEND_BASE_URL", "https://app.flyyv.com")

WATCH_ORIGIN = os.getenv("WATCH_ORIGIN", "LON")
WATCH_DESTINATION = os.getenv("WATCH_DESTINATION", "TLV")
WATCH_START_DATE = os.getenv("WATCH_START_DATE")
WATCH_END_DATE = os.getenv("WATCH_END_DATE")
WATCH_STAY_NIGHTS = int(os.getenv("WATCH_STAY_NIGHTS", "7"))
WATCH_MAX_PRICE = float(os.getenv("WATCH_MAX_PRICE", "720"))

ALERTS_ENABLED = os.getenv("ALERTS_ENABLED", "true").lower() == "true"

# =====================================================================
# SECTION END: ENV, DUFFEL AND EMAIL CONFIG
# =====================================================================

# =====================================================================
# SECTION START: IN MEMORY STORES
# =====================================================================

USER_WALLETS: Dict[str, int] = {}
JOBS: Dict[str, SearchJob] = {}
JOB_RESULTS: Dict[str, List[FlightOption]] = {}

# =====================================================================
# SECTION END: IN MEMORY STORES
# =====================================================================

# =====================================================================
# SECTION START: DUFFEL HELPERS
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
    # Duffel commonly provides cabin at passenger level, but we defensively check multiple shapes
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
    # If any segment is missing cabin, do not guess
    if any(c is None for c in cabins) or not cabins:
        return CabinSummary.UNKNOWN, None

    unique = {c.value for c in cabins if c is not None}
    if len(unique) == 1:
        single = cabins[0]
        return CabinSummary(single.value), single

    highest = max((c for c in cabins if c is not None), key=lambda c: CABIN_RANK.get(c.value, 0))
    return CabinSummary.MIXED, highest

def duffel_post(path: str, payload: dict) -> dict:
    """
    Minimal Duffel POST helper.
    Uses DUFFEL_API_TOKEN, falls back to DUFFEL_ACCESS_TOKEN.
    """
    token = (os.getenv("DUFFEL_API_TOKEN") or os.getenv("DUFFEL_ACCESS_TOKEN") or "").strip()
    if not token:
        raise HTTPException(status_code=500, detail="Duffel token is not configured")

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

    # Try to decode JSON (Duffel returns useful error bodies)
    try:
        data = resp.json()
    except Exception:
        data = {"raw": resp.text}

    # Log response details for debugging (never log the token)
    try:
        import logging
        logger = logging.getLogger("duffel")
        request_id = (
            resp.headers.get("Request-Id")
            or resp.headers.get("Duffel-Request-Id")
            or resp.headers.get("X-Request-Id")
            or resp.headers.get("X-Correlation-Id")
        )
        logger.warning(
            "Duffel POST %s status=%s request_id=%s body=%s",
            path,
            resp.status_code,
            request_id,
            (resp.text or "")[:4000],
        )
    except Exception:
        pass

    if resp.status_code >= 400:
        raise HTTPException(status_code=resp.status_code, detail=data)

    # Most Duffel responses are {"data": ...}
    if isinstance(data, dict) and "data" in data:
        return data["data"]
    return data


def duffel_get(path: str, params: Optional[dict] = None) -> dict:
    """
    Minimal Duffel GET helper.
    Uses DUFFEL_API_TOKEN, falls back to DUFFEL_ACCESS_TOKEN.
    """
    token = (os.getenv("DUFFEL_API_TOKEN") or os.getenv("DUFFEL_ACCESS_TOKEN") or "").strip()
    if not token:
        raise HTTPException(status_code=500, detail="Duffel token is not configured")

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

    # Optional debug log
    try:
        import logging
        logger = logging.getLogger("duffel")
        request_id = (
            resp.headers.get("Request-Id")
            or resp.headers.get("Duffel-Request-Id")
            or resp.headers.get("X-Request-Id")
            or resp.headers.get("X-Correlation-Id")
        )
        logger.warning(
            "Duffel GET %s status=%s request_id=%s body=%s",
            path,
            resp.status_code,
            request_id,
            (resp.text or "")[:4000],
        )
    except Exception:
        pass

    if resp.status_code >= 400:
        raise HTTPException(status_code=resp.status_code, detail=data)

    return data

    if resp.status_code >= 400:
        raise HTTPException(status_code=resp.status_code, detail=data)

    # Duffel responses are usually { "data": {...} }
    if isinstance(data, dict) and "data" in data:
        return data["data"]
    return data


def duffel_list_offers(offer_request_id: str, limit: int = 100) -> List[dict]:
    """
    Returns a list of offers for a given offer_request_id.
    """
    res = duffel_get("/air/offers", params={"offer_request_id": offer_request_id, "limit": int(limit)})
    data = res.get("data") if isinstance(res, dict) else None
    return data or []

def duffel_create_offer_request(slices: List[dict], passengers: List[dict], cabin: str) -> dict:
    """
    Creates a Duffel offer request and returns the JSON response.
    Expects you already have a low-level Duffel POST helper in this file, typically:
      - duffel_post(path: str, payload: dict) -> dict
    If your helper has a different name, change duffel_post below to match it.
    """
    cabin_val = (cabin or "").strip().upper().replace(" ", "_")

    # Duffel expects: economy, premium_economy, business, first
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

    # IMPORTANT: if your project uses a different helper name, update this call.
    return duffel_post("/air/offer_requests", payload)

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
    airline_name = AIRLINE_NAMES.get(
        airline_code,
        owner.get("name", airline_code or "Airline"),
    )
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

            duration_minutes_seg: Optional[int] = None
            if dep_dt and arr_dt:
                try:
                    duration_minutes_seg = int((arr_dt - dep_dt).total_seconds() // 60)
                except Exception:
                    duration_minutes_seg = None

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

            result.append(
                {
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
                }
            )

        return result, total_minutes

    outbound_segments_info, outbound_total_minutes = process_segment_list("outbound", outbound_segments_json)
    return_segments_info, return_total_minutes = process_segment_list("return", return_segments_json)

    # Cabin source of truth: derive from segment cabins, never guess
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
    iso_duration = iso8601_duration(total_duration_minutes)

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
        duration=iso_duration,
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
# SECTION END: DUFFEL HELPERS
# =====================================================================

# =====================================================================
# SECTION START: FILTERING AND BALANCING
# =====================================================================

def apply_filters(options: List[FlightOption], params: SearchParams) -> List[FlightOption]:
    filtered = list(options)

    # Debug: see what we're filtering out
    if filtered:
        stops_dist = Counter(o.stops for o in filtered)
        print(f"[search] apply_filters: input={len(filtered)} stops_dist={dict(stops_dist)} maxPrice={params.maxPrice} stopsFilter={params.stopsFilter}")
    else:
        print("[search] apply_filters: input=0")

    if params.maxPrice is not None and params.maxPrice > 0:
        before = len(filtered)
        filtered = [o for o in filtered if o.price <= params.maxPrice]
        print(f"[search] apply_filters: maxPrice kept={len(filtered)}/{before}")

    if params.stopsFilter:
        before = len(filtered)
        allowed = set(params.stopsFilter)
        if 3 in allowed:
            filtered = [o for o in filtered if (o.stops in allowed or o.stops >= 3)]
        else:
            filtered = [o for o in filtered if o.stops in allowed]
        print(f"[search] apply_filters: stopsFilter kept={len(filtered)}/{before} allowed={sorted(list(allowed))}")

    filtered.sort(key=lambda o: (o.stops, o.price))
    return filtered

def balance_airlines(
    options: List[FlightOption],
    max_total: Optional[int] = None,
) -> List[FlightOption]:
    if not options:
        return []

    sorted_by_price = sorted(options, key=lambda x: x.price)

    if max_total is None or max_total <= 0:
        max_total = len(sorted_by_price)

    actual_total = min(max_total, len(sorted_by_price))

    max_share_percent = get_config_int("MAX_AIRLINE_SHARE_PERCENT", 40)
    if max_share_percent <= 0 or max_share_percent > 100:
        max_share_percent = 40

    airline_counts: Dict[str, int] = defaultdict(int)
    result: List[FlightOption] = []

    airline_buckets: Dict[str, List[FlightOption]] = defaultdict(list)
    for opt in sorted_by_price:
        key = opt.airlineCode or opt.airline
        airline_buckets[key].append(opt)

    unique_airlines = list(airline_buckets.keys())
    num_airlines = max(1, len(unique_airlines))

    base_cap = max(1, (max_share_percent * actual_total) // 100)
    per_airline_cap = max(
        base_cap,
        actual_total // num_airlines if num_airlines else base_cap,
    )

    seen_ids = set()
    for airline_key, bucket in airline_buckets.items():
        if len(result) >= actual_total:
            break

        cheapest_opt = bucket[0]
        if cheapest_opt is None:
            continue

        airline_counts[airline_key] += 1
        result.append(cheapest_opt)
        seen_ids.add(id(cheapest_opt))

    for opt in sorted_by_price:
        if len(result) >= actual_total:
            break

        if id(opt) in seen_ids:
            continue

        key = opt.airlineCode or opt.airline
        if airline_counts[key] >= per_airline_cap:
            continue

        airline_counts[key] += 1
        result.append(opt)
        seen_ids.add(id(opt))

    result.sort(key=lambda x: x.price)
    return result

# =====================================================================
# SECTION END: FILTERING AND BALANCING
# =====================================================================

# =====================================================================
# SECTION START: SHARED SEARCH HELPERS
# =====================================================================

def effective_caps(params: SearchParams) -> Tuple[int, int, int]:
    config_max_pairs = get_config_int("MAX_DATE_PAIRS", 60)
    max_pairs = max(1, min(config_max_pairs, MAX_DATE_PAIRS_HARD))

    requested_per_pair = max(1, params.maxOffersPerPair)
    requested_total = max(1, params.maxOffersTotal)

    config_max_offers_pair = get_config_int("MAX_OFFERS_PER_PAIR", 80)
    config_max_offers_total = get_config_int("MAX_OFFERS_TOTAL", 4000)

    max_offers_pair = max(
        1,
        min(requested_per_pair, config_max_offers_pair, MAX_OFFERS_PER_PAIR_HARD),
    )
    max_offers_total = max(
        1,
        min(requested_total, config_max_offers_total, MAX_OFFERS_TOTAL_HARD),
    )

    return max_pairs, max_offers_pair, max_offers_total
def generate_date_pairs(params, max_pairs: int = 60):
    """
    Build (departure_date, return_date) pairs from the search window and stay rules.

    Uses:
      - params.earliestDeparture, params.latestDeparture
      - params.minStayDays, params.maxStayDays
      - caps to max_pairs and params.maxDatePairs (if present)
    """
    earliest = params.earliestDeparture
    latest = params.latestDeparture
    if not earliest or not latest or earliest > latest:
        return []

    min_stay = max(0, int(getattr(params, "minStayDays", 0) or 0))
    max_stay = max(min_stay, int(getattr(params, "maxStayDays", min_stay) or min_stay))

    # Respect any param cap if it exists
    param_cap = getattr(params, "maxDatePairs", None)
    if isinstance(param_cap, int) and param_cap > 0:
        max_pairs = min(max_pairs, param_cap)

    pairs = []
    dep = earliest
    while dep <= latest and len(pairs) < max_pairs:
        for stay in range(min_stay, max_stay + 1):
            ret = dep + timedelta(days=stay)
            pairs.append((dep, ret))
            if len(pairs) >= max_pairs:
                break
        dep = dep + timedelta(days=1)

    return pairs

def estimate_date_pairs(params: SearchParams) -> int:
    max_pairs, _, _ = effective_caps(params)
    pairs = generate_date_pairs(params, max_pairs=max_pairs)
    return len(pairs)


def apply_global_airline_cap(
    options: List[FlightOption],
    max_share: float = 0.5,
) -> List[FlightOption]:
    if not options:
        print("[search] apply_global_airline_cap: no options, skipping")
        return options

    total = len(options)
    max_per_airline = max(1, int(total * max_share))

    counts: Counter = Counter()
    capped: List[FlightOption] = []

    for opt in options:
        airline = opt.airlineCode or opt.airline or "UNKNOWN"
        if counts[airline] >= max_per_airline:
            continue

        capped.append(opt)
        counts[airline] += 1

    print(
        f"[search] apply_global_airline_cap: input={total}, "
        f"output={len(capped)}, max_per_airline={max_per_airline}, "
        f"airline_counts={dict(counts)}"
    )
    return capped


def fetch_direct_only_offers(
    origin: str,
    destination: str,
    dep_date: date,
    ret_date: date,
    passengers: int,
    cabin: str,
    per_pair_limit: int = 15,
) -> List[FlightOption]:
    if not DUFFEL_ACCESS_TOKEN:
        print("[direct_only] Duffel not configured")
        return []

    slices = [
        {"origin": origin, "destination": destination, "departure_date": dep_date.isoformat()},
        {"origin": destination, "destination": origin, "departure_date": ret_date.isoformat()},
    ]

    pax = [{"type": "adult"} for _ in range(passengers)]

    url = f"{DUFFEL_API_BASE}/air/offer_requests"
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
        
    # duffel_post already raises on non-200s, so if we are here, we have data

    offer_request_id = data.get("id")
    if not offer_request_id:
        print("[direct_only] no offer_request_id returned")
        return []

    try:
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


def run_duffel_scan(params: SearchParams) -> List[FlightOption]:
    print(f"[search] run_duffel_scan START origin={params.origin} dest={params.destination}")

    max_pairs, max_offers_pair, max_offers_total = effective_caps(params)
    print(f"[search] caps max_pairs={max_pairs} max_offers_pair={max_offers_pair} max_offers_total={max_offers_total}")

    date_pairs = generate_date_pairs(params, max_pairs=max_pairs)
    print(f"[search] generated {len(date_pairs)} date pairs")

    max_date_pairs = get_config_int("MAX_DATE_PAIRS_PER_ALERT", 40)
    if max_date_pairs and len(date_pairs) > max_date_pairs:
        print(f"[search] capping date_pairs from {len(date_pairs)} to {max_date_pairs} using MAX_DATE_PAIRS_PER_ALERT")
        date_pairs = date_pairs[:max_date_pairs]

    if not date_pairs:
        print("[search] no date pairs generated, returning empty list")
        return []

    collected_offers: List[Tuple[dict, date, date]] = []
    total_count = 0

    for dep, ret in date_pairs:
        if total_count >= max_offers_total:
            print(f"[search] total_count {total_count} reached max_offers_total {max_offers_total}, stopping")
            break

        print(f"[search] querying Duffel for pair dep={dep} ret={ret} current_total={total_count}")

        slices = [
            {"origin": params.origin, "destination": params.destination, "departure_date": dep.isoformat()},
            {"origin": params.destination, "destination": params.origin, "departure_date": ret.isoformat()},
        ]
        pax = [{"type": "adult"} for _ in range(params.passengers)]

        try:
            offer_request = duffel_create_offer_request(slices, pax, params.cabin)
            offer_request_id = offer_request.get("id")
            if not offer_request_id:
                print("[search] Duffel offer_request returned no id, skipping pair")
                continue

            per_pair_limit = min(max_offers_pair, max_offers_total - total_count)

            # Duffel can return offers inline on offer_request creation.
            # Prefer inline offers to avoid an extra API call and any account limitations on /offers.
            offers_json = offer_request.get("offers") or []
            if offers_json:
                print(f"[search] Duffel offer_request returned {len(offers_json)} inline offers for dep={dep} ret={ret}")
                offers_json = offers_json[:per_pair_limit]
            else:
                print(f"[search] listing offers for request_id={offer_request_id} per_pair_limit={per_pair_limit}")
                offers_json = duffel_list_offers(offer_request_id, limit=per_pair_limit)
        except HTTPException as e:
            print(f"[search] Duffel HTTPException for dep={dep} ret={ret}: {e.detail}")
            continue
        except Exception as e:
            print(f"[search] Unexpected Duffel error for dep={dep} ret={ret}: {e}")
            continue

        print(f"[search] Duffel returned {len(offers_json)} offers for dep={dep} ret={ret}")

        for offer in offers_json:
            collected_offers.append((offer, dep, ret))
            total_count += 1
            if total_count >= max_offers_total:
                print(f"[search] reached max_offers_total={max_offers_total} while collecting offers, breaking inner loop")
                break

    print(f"[search] collected total {len(collected_offers)} offers across all pairs")
    print("[search] starting per date pair mapping, filtering and balancing")

    offers_by_pair: Dict[Tuple[date, date], List[dict]] = defaultdict(list)
    for offer, dep, ret in collected_offers:
        offers_by_pair[(dep, ret)].append(offer)

    all_results: List[FlightOption] = []
    total_added = 0
    hit_global_cap = False

    for dep, ret in date_pairs:
        pair_key = (dep, ret)
        pair_offers = offers_by_pair.get(pair_key, [])
        if not pair_offers:
            print(f"[search] no offers to map for dep={dep} ret={ret}")
            continue

        mapped_pair: List[FlightOption] = [
            map_duffel_offer_to_option(offer, dep, ret, passengers=params.passengers)
            for offer in pair_offers
        ]

        print(f"[search] pair dep={dep} ret={ret}: mapped {len(mapped_pair)} offers")

        filtered_pair = apply_filters(mapped_pair, params)
        print(f"[search] pair dep={dep} ret={ret}: filtered down to {len(filtered_pair)} offers")

        if not filtered_pair:
            print(f"[search] pair dep={dep} ret={ret}: no offers after filters")
            continue

        airline_counts_pair = Counter(opt.airlineCode or opt.airline for opt in filtered_pair)
        print(f"[search] airline mix before balance for dep={dep} ret={ret}: {dict(airline_counts_pair)}")

        if len(filtered_pair) > max_offers_pair:
            print(f"[search] pair dep={dep} ret={ret}: capping offers from {len(filtered_pair)} to {max_offers_pair} using max_offers_pair")
            filtered_pair = filtered_pair[:max_offers_pair]

        balanced_pair = balance_airlines(filtered_pair, max_total=max_offers_pair)
        print(f"[search] pair dep={dep} ret={ret}: balance_airlines returned {len(balanced_pair)} offers")

        for opt in balanced_pair:
            if total_added >= max_offers_total:
                print(f"[search] global cap reached while adding dep={dep} ret={ret}, max_offers_total={max_offers_total}")
                hit_global_cap = True
                break
            all_results.append(opt)
            total_added += 1

        if hit_global_cap:
            break

    print(f"[search] run_duffel_scan DONE, returning {len(all_results)} offers from {len(date_pairs)} date pairs, hit_global_cap={hit_global_cap}")
    return all_results

# =====================================================================
# SECTION END: SHARED SEARCH HELPERS
# =====================================================================

# =====================================================================
# SECTION START: ASYNC JOB RUNNER
# =====================================================================

def process_date_pair_offers(
    params: SearchParams,
    dep: date,
    ret: date,
    max_offers_pair: int,
) -> List[FlightOption]:
    """
    Fetch offers for one (dep, ret) pair:
      1) Duffel offers (limit=max_offers_pair)
      2) Direct-only offers (small extra pool)
      3) Merge direct-only into the pair pool without duplicates
    """
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
        map_duffel_offer_to_option(offer, dep, ret, passengers=params.passengers)
        for offer in offers_json
    ]

    # Pull a small "direct-only" pool and merge in, this improves perceived trust and speed for users
    try:
        direct_options = fetch_direct_only_offers(
            origin=params.origin,
            destination=params.destination,
            dep_date=dep,
            ret_date=ret,
            passengers=params.passengers,
            cabin=params.cabin,
            per_pair_limit=15,
        )
    except Exception as e:
        print(f"[PAIR {dep} -> {ret}] direct_only error: {e}")
        direct_options = []

    if direct_options:
        seen = set()
        for opt in batch_mapped:
            key = (
                opt.airlineCode or opt.airline,
                opt.departureDate,
                opt.returnDate,
                getattr(opt, "stops", getattr(opt, "numStops", None)),
                opt.originAirport,
                opt.destinationAirport,
            )
            seen.add(key)

        added = 0
        for opt in direct_options:
            key = (
                opt.airlineCode or opt.airline,
                opt.departureDate,
                opt.returnDate,
                getattr(opt, "stops", getattr(opt, "numStops", None)),
                opt.originAirport,
                opt.destinationAirport,
            )
            if key in seen:
                continue
            batch_mapped.append(opt)
            seen.add(key)
            added += 1

        print(f"[PAIR {dep} -> {ret}] merged {added} direct-only offers, total now {len(batch_mapped)}")

    return batch_mapped


def run_search_job(job_id: str):
    """
    Async job runner:
      - Generates date pairs
      - Fetches offers per pair in parallel
      - Applies per-pair curation (max 20, direct quota, per-pair airline cap)
      - Merges curated results into global results with global caps
    """
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

        if total_pairs == 0:
            job.status = JobStatus.COMPLETED
            job.updated_at = datetime.utcnow()
            JOBS[job_id] = job
            print(f"[JOB {job_id}] No date pairs, completed with 0 options")
            return

        total_count = 0
        parallel_workers = get_config_int("PARALLEL_WORKERS", PARALLEL_WORKERS)
        parallel_workers = max(1, min(parallel_workers, 16))

        batch_timeout_seconds = 120

        # Read optional tuning knobs from admin_config (Directus).
        # Defaults are safe and match your plan.
        direct_quota_pct_raw = get_config_str("DIRECT_QUOTA_PCT", None)
        per_pair_cap_pct_raw = get_config_str("PER_PAIR_AIRLINE_CAP_PCT", None)

        try:
            direct_quota_pct = float(direct_quota_pct_raw) if direct_quota_pct_raw else 0.3
        except Exception:
            direct_quota_pct = 0.3

        try:
            per_pair_airline_cap_pct = float(per_pair_cap_pct_raw) if per_pair_cap_pct_raw else 0.3
        except Exception:
            per_pair_airline_cap_pct = 0.3

        direct_quota_pct = max(0.0, min(direct_quota_pct, 1.0))
        per_pair_airline_cap_pct = max(0.0, min(per_pair_airline_cap_pct, 1.0))

        pair_cap = max(1, int(max_offers_pair))

        # Convert pct to absolute per-airline cap inside each date-pair.
        if per_pair_airline_cap_pct <= 0.0:
            per_airline_cap = pair_cap
        else:
            per_airline_cap = int(round(pair_cap * per_pair_airline_cap_pct))
            per_airline_cap = max(1, min(per_airline_cap, pair_cap))

        direct_slots = int(round(pair_cap * direct_quota_pct))
        direct_slots = max(0, min(direct_slots, pair_cap))
        non_direct_slots = pair_cap - direct_slots

        def _is_direct(opt: FlightOption) -> bool:
            stops_val = getattr(opt, "stops", getattr(opt, "numStops", None))
            if stops_val is None:
                return False
            try:
                return int(stops_val) == 0
            except Exception:
                return False

        def _airline_key(opt: FlightOption) -> str:
            v = getattr(opt, "airlineCode", None) or getattr(opt, "airline", None)
            if v:
                return str(v)
            return "UNKNOWN"

        def _take_with_cap(candidates: List[FlightOption], limit: int) -> List[FlightOption]:
            picked: List[FlightOption] = []
            counts: Dict[str, int] = {}
            for opt in candidates:
                if len(picked) >= limit:
                    break
                a = _airline_key(opt)
                if counts.get(a, 0) >= per_airline_cap:
                    continue
                picked.append(opt)
                counts[a] = counts.get(a, 0) + 1
            return picked

        with ThreadPoolExecutor(max_workers=parallel_workers) as executor:
            for batch_start in range(0, total_pairs, parallel_workers):
                if total_count >= max_offers_total:
                    print(f"[JOB {job_id}] Reached max_offers_total before batch, stopping")
                    break

                batch_pairs = date_pairs[batch_start : batch_start + parallel_workers]
                futures = {
                    executor.submit(
                        process_date_pair_offers,
                        job.params,
                        dep,
                        ret,
                        max_offers_pair,
                    ): (dep, ret)
                    for dep, ret in batch_pairs
                }

                try:
                    for future in as_completed(futures, timeout=batch_timeout_seconds):
                        dep, ret = futures[future]

                        job.processed_pairs += 1
                        job.updated_at = datetime.utcnow()
                        JOBS[job_id] = job

                        try:
                            batch_mapped = future.result()
                        except Exception as e:
                            print(f"[JOB {job_id}] Future error for pair {dep} -> {ret}: {e}")
                            continue

                        if not batch_mapped:
                            continue

                        # 1) Pair-level filtering only
                        filtered_pair = apply_filters(batch_mapped, job.params)
                        if not filtered_pair:
                            continue

                        # 2) Pair-level curation to pair_cap with direct quota and per-pair airline cap
                        pair_direct = [o for o in filtered_pair if _is_direct(o)]
                        pair_non_direct = [o for o in filtered_pair if not _is_direct(o)]

                        picked_direct = _take_with_cap(pair_direct, direct_slots)

                        # Roll unused direct slots into non-direct
                        remaining_for_non_direct = non_direct_slots + max(0, direct_slots - len(picked_direct))
                        picked_non_direct = _take_with_cap(pair_non_direct, remaining_for_non_direct)

                        curated_pair = picked_direct + picked_non_direct

                        # Top up if we still have room (airline cap too strict)
                        if len(curated_pair) < pair_cap:
                            seen_ids = {id(o) for o in curated_pair}
                            for opt in filtered_pair:
                                if len(curated_pair) >= pair_cap:
                                    break
                                if id(opt) in seen_ids:
                                    continue
                                curated_pair.append(opt)
                                seen_ids.add(id(opt))

                        # Final hard cap
                        if len(curated_pair) > pair_cap:
                            curated_pair = curated_pair[:pair_cap]

                        # 3) Optional: keep your existing balance step (still inside this date-pair only)
                        balanced_pair = balance_airlines(curated_pair, max_total=pair_cap)
                        if not balanced_pair:
                            continue

                        # Keep your existing "stricter per-pair cap" behavior.
                        # If apply_global_airline_cap is truly global-only in semantics, it still works
                        # as a final safeguard inside this curated list.
                        balanced_pair = apply_global_airline_cap(balanced_pair, max_share=per_pair_airline_cap_pct)

                        try:
                            direct_taken = sum(1 for o in balanced_pair if _is_direct(o))
                            uniq_airlines = len({_airline_key(o) for o in balanced_pair})
                            print(
                                f"[PAIR {dep} -> {ret}] curated pair_cap={pair_cap}, "
                                f"direct_slots={direct_slots}, direct_taken={direct_taken}, "
                                f"returned={len(balanced_pair)}, uniq_airlines={uniq_airlines}"
                            )
                        except Exception as _e:
                            print(f"[PAIR {dep} -> {ret}] curated debug failed: {_e}")

                        # 4) Merge into global results with global caps as before
                        current_results = JOB_RESULTS.get(job_id, [])
                        remaining_slots = max_offers_total - len(current_results)
                        if remaining_slots <= 0:
                            total_count = len(current_results)
                            break

                        if len(balanced_pair) > remaining_slots:
                            balanced_pair = balanced_pair[:remaining_slots]

                        merged = current_results + balanced_pair
                        merged = apply_global_airline_cap(merged, max_share=0.5)

                        JOB_RESULTS[job_id] = merged
                        total_count = len(merged)

                        if total_count >= max_offers_total:
                            break

                except Exception as e:
                    error_msg = (
                        f"Timed out or failed waiting for batch Duffel responses after "
                        f"{batch_timeout_seconds} seconds: {e}"
                    )
                    print(f"[JOB {job_id}] {error_msg}")
                    job.status = JobStatus.FAILED
                    job.error = error_msg
                    job.updated_at = datetime.utcnow()
                    JOBS[job_id] = job
                    return

                if total_count >= max_offers_total:
                    break

        final_results = JOB_RESULTS.get(job_id, [])
        final_results = apply_global_airline_cap(final_results, max_share=0.5)
        JOB_RESULTS[job_id] = final_results

        job.status = JobStatus.COMPLETED
        job.updated_at = datetime.utcnow()
        JOBS[job_id] = job
        print(f"[JOB {job_id}] Completed with {len(final_results)} options")

    except Exception as e:
        job.status = JobStatus.FAILED
        job.error = str(e)
        job.updated_at = datetime.utcnow()
        JOBS[job_id] = job
        print(f"[JOB {job_id}] FAILED: {e}")

# =====================================================================
# SECTION END: ASYNC JOB RUNNER
# =====================================================================

# =====================================================================
# SECTION START: PRICE WATCH HELPERS
# =====================================================================

def run_price_watch() -> Dict[str, Any]:
    if not WATCH_START_DATE or not WATCH_END_DATE:
        raise HTTPException(
            status_code=500,
            detail="WATCH_START_DATE and WATCH_END_DATE must be configured in YYYY-MM-DD format",
        )

    try:
        start = datetime.strptime(WATCH_START_DATE, "%Y-%m-%d").date()
        end = datetime.strptime(WATCH_END_DATE, "%Y-%m-%d").date()
    except ValueError:
        raise HTTPException(
            status_code=500,
            detail="WATCH_START_DATE or WATCH_END_DATE invalid format, expected YYYY-MM-DD",
        )

    if end < start:
        raise HTTPException(
            status_code=500,
            detail="WATCH_END_DATE must be on or after WATCH_START_DATE",
        )

    params = SearchParams(
        origin=WATCH_ORIGIN,
        destination=WATCH_DESTINATION,
        earliestDeparture=start,
        latestDeparture=end,
        minStayDays=WATCH_STAY_NIGHTS,
        maxStayDays=WATCH_STAY_NIGHTS,
        maxPrice=None,
        cabin="BUSINESS",
        passengers=1,
        stopsFilter=None,
    )

    watched_pairs = generate_date_pairs(params, max_pairs=365)
    options = run_duffel_scan(params)

    scanned_pairs: List[Tuple[str, str]] = sorted({(opt.departureDate, opt.returnDate) for opt in options})
    if scanned_pairs:
        last_scanned_dep, last_scanned_ret = scanned_pairs[-1]
    else:
        last_scanned_dep = None
        last_scanned_ret = None

    grouped: Dict[Tuple[str, str], List[FlightOption]] = defaultdict(list)
    for opt in options:
        key = (opt.departureDate, opt.returnDate)
        grouped[key].append(opt)

    pairs_summary: List[Dict[str, Any]] = []
    any_under = False

    for dep, ret in watched_pairs:
        dep_str = dep.isoformat()
        ret_str = ret.isoformat()

        if last_scanned_dep is not None and dep_str > last_scanned_dep:
            break

        flights = grouped.get((dep_str, ret_str), [])

        if not flights:
            status = "no_data"
            cheapest_price = None
            cheapest_currency = None
            cheapest_airline = None
            flights_under: List[Dict[str, Any]] = []
            total_flights = 0
            min_price = None
            max_price = None
        else:
            flights_sorted = sorted(flights, key=lambda f: f.price)
            total_flights = len(flights_sorted)
            min_price = flights_sorted[0].price
            max_price = flights_sorted[-1].price

            cheapest = flights_sorted[0]
            cheapest_price = cheapest.price
            cheapest_currency = cheapest.currency
            cheapest_airline = cheapest.airline

            if cheapest_price <= WATCH_MAX_PRICE:
                status = "under_threshold"
                any_under = True
            else:
                status = "above_threshold"

            flights_under = []
            if status == "under_threshold":
                # build_flyyv_link exists in alerts_email.py, used there.
                # Keeping the original behavior here, you can adapt later.
                flights_under.append(
                    {
                        "airline": cheapest.airline,
                        "airlineCode": cheapest.airlineCode,
                        "price": cheapest.price,
                        "currency": cheapest.currency,
                        "origin": cheapest.origin,
                        "destination": cheapest.destination,
                        "departureDate": cheapest.departureDate,
                        "returnDate": cheapest.returnDate,
                        "flyyvLink": "",
                        "airlineUrl": cheapest.url or "",
                    }
                )

        pairs_summary.append(
            {
                "departureDate": dep_str,
                "returnDate": ret_str,
                "status": status,
                "cheapestPrice": cheapest_price,
                "cheapestCurrency": cheapest_currency,
                "cheapestAirline": cheapest_airline,
                "totalFlights": total_flights,
                "minPrice": min_price,
                "maxPrice": max_price,
                "flightsUnderThreshold": flights_under,
            }
        )

    return {
        "origin": WATCH_ORIGIN,
        "destination": WATCH_DESTINATION,
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "cabin": params.cabin,
        "passengers": params.passengers,
        "stay_nights": WATCH_STAY_NIGHTS,
        "max_price": WATCH_MAX_PRICE,
        "any_under_threshold": any_under,
        "last_scanned_departure": last_scanned_dep,
        "last_scanned_return": last_scanned_ret,
        "pairs": pairs_summary,
    }

# =====================================================================
# SECTION END: PRICE WATCH HELPERS
# =====================================================================

# =====================================================================
# SECTION START: ALERT ENGINE HELPERS
# =====================================================================

def _derive_alert_passengers(alert: Any) -> int:
    """
    Backwards compatible passenger resolution.
    Supports DB models that do not yet have a passengers column.
    """
    value = getattr(alert, "passengers", None)
    if value is None:
        value = getattr(alert, "number_of_passengers", None)
    try:
        value_int = int(value) if value is not None else 1
    except Exception:
        value_int = 1
    return max(1, value_int)


def build_search_params_for_alert(alert: Alert) -> SearchParams:
    dep_start = alert.departure_start
    dep_end = alert.departure_end or alert.departure_start

    if alert.return_start and alert.return_end:
        min_stay = max(1, (alert.return_start - dep_start).days)
        max_stay = max(min_stay, (alert.return_end - dep_start).days)
    else:
        min_stay = 1
        max_stay = 21

    pax = _derive_alert_passengers(alert)

    return SearchParams(
        origin=alert.origin,
        destination=alert.destination,
        earliestDeparture=dep_start,
        latestDeparture=dep_end,
        minStayDays=min_stay,
        maxStayDays=max_stay,
        maxPrice=None,
        cabin=alert.cabin or "BUSINESS",
        passengers=pax,
        stopsFilter=None,
        maxOffersPerPair=120,
        maxOffersTotal=1200,
    )


def process_alert(alert: Alert, db: Session) -> None:
    now = datetime.utcnow()
    print(
        f"[alerts] process_alert START id={alert.id} "
        f"email={alert.user_email} type={alert.alert_type} mode={alert.mode}"
    )

    user = db.query(AppUser).filter(AppUser.email == alert.user_email).first()

    if not user:
        db.add(AlertRun(
            id=str(uuid4()),
            alert_id=alert.id,
            run_at=now,
            price_found=None,
            sent=False,
            reason="no_user_for_alert",
        ))
        alert.last_run_at = now
        alert.updated_at = now
        db.commit()
        return

    if not should_send_alert(db, user):
        db.add(AlertRun(
            id=str(uuid4()),
            alert_id=alert.id,
            run_at=now,
            price_found=None,
            sent=False,
            reason="alerts_disabled",
        ))
        alert.last_run_at = now
        alert.updated_at = now
        db.commit()
        return

    params = build_search_params_for_alert(alert)
    if params is None:
        raise RuntimeError("build_search_params_for_alert returned None")

    # For under-price alerts, do not apply maxPrice during the scan itself.
    if alert.max_price is not None:
        try:
            scan_params = params.model_copy(update={"maxPrice": None})  # Pydantic v2
        except Exception:
            try:
                scan_params = params.copy(update={"maxPrice": None})  # Pydantic v1
            except Exception:
                from copy import deepcopy
                scan_params = deepcopy(params)
                scan_params.maxPrice = None
    else:
        scan_params = params

    options = run_duffel_scan(scan_params)
    print(f"[alerts] scan complete alert_id={alert.id} options_count={len(options)} scan_maxPrice={getattr(scan_params, 'maxPrice', None)}")

    if not options:
        db.add(AlertRun(
            id=str(uuid4()),
            alert_id=alert.id,
            run_at=now,
            price_found=None,
            sent=False,
            reason="no_results_scan_empty",

        ))
        alert.last_run_at = now
        alert.updated_at = now
        db.commit()
        return

    options_sorted = sorted(options, key=lambda o: o.price)
    cheapest = options_sorted[0]
    current_price = int(cheapest.price)

        # Stored best price = lowest price_found in AlertRun history
    best_run = (
        db.query(AlertRun)
        .filter(AlertRun.alert_id == alert.id)
        .filter(AlertRun.price_found.isnot(None))
        .order_by(AlertRun.price_found.asc())
        .first()
    )
    stored_best_price = int(best_run.price_found) if best_run and best_run.price_found is not None else None

    should_send = False
    send_reason = None

    # Map legacy types, then override to "under_price" if a threshold is set
    effective_type = alert.alert_type
    if effective_type == "price_change":
        effective_type = "new_best"
    elif effective_type == "scheduled_3x":
        effective_type = "summary"

    if alert.max_price is not None:
        effective_type = "under_price"

    stored_best_price = alert.last_price

    # Deterministic email rules
    if effective_type == "under_price":
        if alert.max_price is not None and current_price <= int(alert.max_price):
            should_send = True
            send_reason = "under_price"
        else:
            send_reason = "not_under_price"

    elif effective_type == "new_best":
        if stored_best_price is None:
            should_send = True
            send_reason = "first_best"
        elif current_price < stored_best_price:
            should_send = True
            send_reason = "new_best"
        else:
            send_reason = "not_new_best"

    elif effective_type == "summary":
        should_send = True
        send_reason = "summary"

    else:
        should_send = False
        send_reason = "unknown_alert_type"

    sent_flag = False

    if should_send:
        try:
            if alert.mode == "smart":
                send_smart_alert_email(alert, options_sorted, params)
            else:
                send_alert_email_for_alert(alert, cheapest, params)
            sent_flag = True
        except Exception as e:
            print(f"[alerts] Failed to send email for alert {alert.id}: {e}")
            sent_flag = False
            send_reason = "email_failed"

    db.add(AlertRun(
        id=str(uuid4()),
        alert_id=alert.id,
        run_at=now,
        price_found=current_price,
        sent=sent_flag,
        reason=send_reason,
    ))

    if stored_best_price is None or current_price < stored_best_price:
        alert.last_price = current_price

        alert.last_run_at = now
        alert.updated_at = now

    if sent_flag:
        alert.times_sent = (alert.times_sent or 0) + 1
        alert.last_notified_at = now
        alert.last_notified_price = current_price

    db.commit()


def run_all_alerts_cycle() -> None:
    if not master_alerts_enabled():
        print("[alerts] ALERTS_ENABLED is false, skipping alerts cycle")
        return

    if not DUFFEL_ACCESS_TOKEN:
        print("[alerts] DUFFEL_ACCESS_TOKEN not set, skipping alerts cycle")
        return

    if not (SMTP_USERNAME and SMTP_PASSWORD and ALERT_FROM_EMAIL):
        print("[alerts] SMTP not fully configured, skipping alerts cycle")
        return

    db = SessionLocal()
    try:
        if not alerts_globally_enabled(db):
            print("[alerts] Global alerts disabled in admin_config, skipping alerts cycle")
            return

        alerts = db.query(Alert).filter(Alert.is_active == True).all()  # noqa: E712
        print(f"[alerts] Running alerts cycle for {len(alerts)} alerts")

        import traceback

        for alert in alerts:
            try:
                process_alert(alert, db)
            except Exception as e:
                print(f"[alerts] Error processing alert {alert.id}: {e}")
                traceback.print_exc()

    finally:
        db.close()

# =====================================================================
# SECTION END: ALERT ENGINE HELPERS
# =====================================================================

# =====================================================================
# SECTION START: EMAIL HELPERS
# =====================================================================

def send_test_alert_email() -> None:
    if not (SMTP_USERNAME and SMTP_PASSWORD and ALERT_TO_EMAIL):
        raise HTTPException(status_code=500, detail="SMTP settings are not fully configured on the server")

    msg = EmailMessage()
    msg["Subject"] = "Flyyv test alert email"
    msg["From"] = ALERT_FROM_EMAIL
    msg["To"] = ALERT_TO_EMAIL
    msg.set_content("This is a test Flyyv alert sent via SMTP2Go.\n\nIf you are reading this, SMTP is working.")

    try:
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
            server.starttls()
            server.login(SMTP_USERNAME, SMTP_PASSWORD)
            server.send_message(msg)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to send test email: {e}")

# =====================================================================
# SECTION END: EMAIL HELPERS
# =====================================================================

# =====================================================================
# SECTION START: ROOT, HEALTH AND ROUTES
# =====================================================================

@app.get("/")
def home():
    return {"message": "Flyyv backend is running"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/routes")
def list_routes():
    return [route.path for route in app.routes]


@app.get("/test-email-alert")
def test_email_alert():
    send_test_alert_email()
    return {"detail": "Test alert email sent"}


@app.get("/test-email-confirmation")
def test_email_confirmation(email: str = ALERT_TO_EMAIL, flex: int = 0):
    class DummyAlert:
        user_email = email
        origin = "LON"
        destination = "TLV"
        cabin = "BUSINESS"
        departure_start = datetime.utcnow().date()
        departure_end = (datetime.utcnow() + timedelta(days=30)).date()
        return_start = datetime.utcnow().date() + timedelta(days=7)
        return_end = datetime.utcnow().date() + timedelta(days=14)
        mode = "smart" if flex == 1 else "single"
        search_mode = "flexible" if flex == 1 else "single"
        passengers = 1

    send_alert_confirmation_email(DummyAlert())
    return {"detail": f"Test confirmation email sent to {email}, flex={flex}"}

@app.get("/test-email-smart-alert")
def test_email_smart_alert(email: str = ALERT_TO_EMAIL):
    print(f"[test-email-smart-alert] START to={email}")

    class DummyAlert:
        user_email = email
        origin = "LON"
        destination = "TLV"
        cabin = "BUSINESS"
        max_price = 2200
        departure_start = datetime.utcnow().date()
        departure_end = (datetime.utcnow() + timedelta(days=30)).date()
        return_start = datetime.utcnow().date() + timedelta(days=7)
        return_end = (datetime.utcnow() + timedelta(days=7)).date()
        mode = "smart"
        search_mode = "flexible"
        passengers = 1

    class DummyOption:
        def __init__(self, dep, ret, price, airline):
            self.departureDate = dep
            self.returnDate = ret
            self.price = price
            self.airline = airline

    options = [
        DummyOption(
            (datetime.utcnow() + timedelta(days=5)).date().isoformat(),
            (datetime.utcnow() + timedelta(days=12)).date().isoformat(),
            1890,
            "British Airways",
        ),
        DummyOption(
            (datetime.utcnow() + timedelta(days=8)).date().isoformat(),
            (datetime.utcnow() + timedelta(days=15)).date().isoformat(),
            2010,
            "EL AL",
        ),
        DummyOption(
            (datetime.utcnow() + timedelta(days=11)).date().isoformat(),
            (datetime.utcnow() + timedelta(days=18)).date().isoformat(),
            2140,
            "Lufthansa",
        ),
    ]

    class DummyParams:
        origin = "LON"
        destination = "TLV"
        cabin = "BUSINESS"
        passengers = 1
        earliestDeparture = datetime.utcnow()
        latestDeparture = datetime.utcnow() + timedelta(days=30)
        search_mode = "flexible"

    send_smart_alert_email(DummyAlert(), options, DummyParams())

    print(f"[test-email-smart-alert] SENT OK to={email}")
    return {"detail": f"Test smart alert email sent to {email}"}

@app.get("/trigger-daily-alert")
def trigger_daily_alert(background_tasks: BackgroundTasks):
    if not ALERTS_ENABLED:
        return {"detail": "Alerts are currently disabled via environment"}

    system_enabled = get_config_bool("ALERTS_SYSTEM_ENABLED", True)
    if not system_enabled:
        return {"detail": "Alerts are currently disabled in admin config"}

    background_tasks.add_task(run_all_alerts_cycle)
    return {"detail": "Alerts cycle queued"}

# =====================================================================
# SECTION END: ROOT, HEALTH AND ROUTES
# =====================================================================


# =====================================================================
# SECTION START: MAIN SEARCH ROUTES
# =====================================================================

@app.post("/search-business")
def search_business(params: SearchParams, background_tasks: BackgroundTasks):
    if not DUFFEL_ACCESS_TOKEN:
        return {"status": "error", "source": "duffel_not_configured", "options": []}

    max_passengers = get_config_int("MAX_PASSENGERS", 4)
    if params.passengers > max_passengers:
        params.passengers = max_passengers

    default_cabin = get_config_str("DEFAULT_CABIN", "BUSINESS") or "BUSINESS"
    if not params.cabin:
        params.cabin = default_cabin

    estimated_pairs = estimate_date_pairs(params)

    if estimated_pairs <= 1:
        options = run_duffel_scan(params)
        options = apply_global_airline_cap(options, max_share=0.5)
        return {"status": "ok", "mode": "sync", "source": "duffel", "options": [o.dict() for o in options]}

    job_id = str(uuid4())
    job = SearchJob(
        id=job_id,
        status=JobStatus.PENDING,
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
        params=params,
        total_pairs=0,
        processed_pairs=0,
    )
    JOBS[job_id] = job
    JOB_RESULTS[job_id] = []
    background_tasks.add_task(run_search_job, job_id)

    return {"status": "ok", "mode": "async", "jobId": job_id, "message": "Search started"}

# =====================================================================
# SECTION END: MAIN SEARCH ROUTES
# =====================================================================


# =====================================================================
# SECTION START: SEARCH STATUS AND RESULTS ROUTES
# =====================================================================

@app.get("/search-status/{job_id}", response_model=SearchStatusResponse)
def get_search_status(job_id: str, preview_limit: int = 20):
    job = JOBS.get(job_id)

    if not job:
        return SearchStatusResponse(
            jobId=job_id,
            status=JobStatus.PENDING,
            processedPairs=0,
            totalPairs=0,
            progress=0.0,
            error="Job not found in memory yet",
            previewCount=0,
            previewOptions=[],
        )

    options = JOB_RESULTS.get(job_id, [])
    preview = options[:preview_limit] if preview_limit > 0 else []

    total_pairs = job.total_pairs or 0
    processed_pairs = job.processed_pairs or 0
    progress = float(processed_pairs) / float(total_pairs) if total_pairs > 0 else 0.0

    return SearchStatusResponse(
        jobId=job.id,
        status=job.status,
        processedPairs=processed_pairs,
        totalPairs=total_pairs,
        progress=progress,
        error=job.error,
        previewCount=len(preview),
        previewOptions=preview,
    )


@app.get("/search-results/{job_id}", response_model=SearchResultsResponse)
def get_search_results(job_id: str, offset: int = 0, limit: int = 50):
    job = JOBS.get(job_id)
    if not job:
        return SearchResultsResponse(
            jobId=job_id,
            status=JobStatus.PENDING,
            totalResults=0,
            offset=0,
            limit=limit,
            options=[],
        )

    options = JOB_RESULTS.get(job_id, [])

    offset = max(0, offset)
    limit = max(1, min(limit, 600))
    end = min(offset + limit, len(options))
    slice_ = options[offset:end]

    return SearchResultsResponse(
        jobId=job.id,
        status=job.status,
        totalResults=len(options),
        offset=offset,
        limit=limit,
        options=slice_,
    )

# =====================================================================
# SECTION END: SEARCH STATUS AND RESULTS ROUTES
# =====================================================================


# =====================================================================
# SECTION START: ADMIN CREDITS ENDPOINT
# =====================================================================

@app.post("/admin/add-credits")
def admin_add_credits(
    payload: CreditUpdateRequest,
    x_admin_token: str = Header(None, alias="X-Admin-Token"),
):
    received = (x_admin_token or "").strip()
    expected = (ADMIN_API_TOKEN or "").strip()

    if received.lower().startswith("bearer "):
        received = received[7:].strip()

    if expected == "":
        raise HTTPException(status_code=500, detail="Admin token not configured")

    if received != expected:
        raise HTTPException(status_code=401, detail="Invalid admin token")

    change_amount = (
        payload.delta
        if payload.delta is not None
        else payload.amount
        if payload.amount is not None
        else payload.creditAmount
        if payload.creditAmount is not None
        else payload.value
    )

    if change_amount is None:
        raise HTTPException(
            status_code=400,
            detail="Missing credit amount. Expected one of: amount, delta, creditAmount, value.",
        )

    current = USER_WALLETS.get(payload.userId, 0)
    new_balance = max(0, current + change_amount)
    USER_WALLETS[payload.userId] = new_balance

    return {"userId": payload.userId, "newBalance": new_balance}

# =====================================================================
# SECTION END: ADMIN CREDITS ENDPOINT
# =====================================================================


# =====================================================================
# SECTION START: PUBLIC CONFIG, USER SYNC, PROFILE
# =====================================================================

@app.get("/public-config", response_model=PublicConfig)
def public_config():
    max_window = get_config_int("MAX_DEPARTURE_WINDOW_DAYS", 60)
    max_stay = get_config_int("MAX_STAY_NIGHTS", 30)
    min_stay = get_config_int("MIN_STAY_NIGHTS", 1)
    max_passengers = get_config_int("MAX_PASSENGERS", 4)

    return PublicConfig(
        maxDepartureWindowDays=max_window,
        maxStayNights=max_stay,
        minStayNights=min_stay,
        maxPassengers=max_passengers,
    )


@app.post("/user-sync")
def user_sync(payload: UserSyncPayload):
    db = SessionLocal()
    try:
        user = db.query(AppUser).filter(AppUser.external_id == payload.external_id).first()

        if user is None:
            user = AppUser(
                external_id=payload.external_id,
                email=payload.email,
                first_name=payload.first_name,
                last_name=payload.last_name,
                country=payload.country,
                marketing_consent=payload.marketing_consent,
                source=payload.source or "base44",
            )
            db.add(user)
        else:
            user.email = payload.email
            user.first_name = payload.first_name
            user.last_name = payload.last_name
            user.country = payload.country
            user.marketing_consent = payload.marketing_consent
            user.source = payload.source or user.source

        db.commit()
        db.refresh(user)

        return {"status": "ok", "id": user.id}
    finally:
        db.close()


@app.get("/profile", response_model=ProfileResponse)
def get_profile(x_user_id: str = Header(..., alias="X-User-Id")):
    wallet_balance = USER_WALLETS.get(x_user_id, 0)

    db = SessionLocal()
    try:
        app_user = db.query(AppUser).filter(AppUser.external_id == x_user_id).first()
    finally:
        db.close()

    if app_user:
        name_parts: List[str] = []
        if app_user.first_name:
            name_parts.append(app_user.first_name)
        if app_user.last_name:
            name_parts.append(app_user.last_name)
        name = " ".join(name_parts) or "Flyyv user"
        email = app_user.email
    else:
        name = "Flyyv user"
        email = None

    profile_user = ProfileUser(id=x_user_id, name=name, email=email, plan="Free")

    subscription = SubscriptionInfo(plan="Flyyv Free", billingCycle=None, renewalDate=None, monthlyCredits=0)

    wallet = WalletInfo(balance=wallet_balance, currency="credits")

    return ProfileResponse(user=profile_user, subscription=subscription, wallet=wallet)

# =====================================================================
# SECTION END: PUBLIC CONFIG, USER SYNC, PROFILE
# =====================================================================


# =====================================================================
# SECTION START: ALERT ROUTES
# =====================================================================

@app.post("/alerts", response_model=AlertOut)
def create_alert(payload: AlertCreate):
    db = SessionLocal()
    try:
        alert_id = str(uuid4())
        now = datetime.utcnow()

        search_mode_value = (payload.search_mode or "flexible").strip().lower()
        if search_mode_value not in ("flexible", "fixed"):
            raise HTTPException(status_code=400, detail="Invalid search_mode")

        mode_value = (payload.mode or "").strip().lower()
        if search_mode_value == "flexible":
            mode_value = "smart"
        elif mode_value not in ("smart", "single"):
            mode_value = "single"

        max_passengers = get_config_int("MAX_PASSENGERS", 4)
        pax = max(1, int(payload.passengers or 1))
        if pax > max_passengers:
            pax = max_passengers

        alert = Alert(
            id=alert_id,
            user_email=payload.email,
            origin=payload.origin,
            destination=payload.destination,
            cabin=payload.cabin,
            search_mode=search_mode_value,
            departure_start=payload.departure_start,
            departure_end=payload.departure_end,
            return_start=payload.return_start,
            return_end=payload.return_end,
            alert_type=("under_price" if payload.max_price is not None else payload.alert_type),
            max_price=payload.max_price,
            mode=mode_value,
            last_price=None,
            last_run_at=None,
            times_sent=0,
            is_active=True,
            created_at=now,
            updated_at=now,
        )

        if hasattr(alert, "passengers"):
            alert.passengers = pax

        db.add(alert)
        db.commit()
        db.refresh(alert)

        try:
            send_alert_confirmation_email(alert)
        except Exception:
            pass

        return AlertOut(
            id=alert.id,
            email=alert.user_email,
            origin=alert.origin,
            destination=alert.destination,
            cabin=alert.cabin,
            search_mode=alert.search_mode,
            departure_start=alert.departure_start,
            departure_end=alert.departure_end,
            return_start=alert.return_start,
            return_end=alert.return_end,
            alert_type=alert.alert_type,
            max_price=alert.max_price,
            mode=alert.mode,
            passengers=_derive_alert_passengers(alert),
            times_sent=alert.times_sent,
            is_active=alert.is_active,
            last_price=alert.last_price,
            best_price=None,
            last_run_at=alert.last_run_at,
            last_notified_at=None,
            last_notified_price=None,
            created_at=alert.created_at,
            updated_at=alert.updated_at,
        )
    finally:
        db.close()


@app.get("/alerts", response_model=List[AlertOut])
def get_alerts(
    email: Optional[str] = None,
    x_user_id: Optional[str] = Header(None, alias="X-User-Id"),
    include_inactive: bool = False,
):
    db = SessionLocal()
    try:
        resolved_email: Optional[str] = None

        if email:
            resolved_email = email
        elif x_user_id:
            app_user = db.query(AppUser).filter(AppUser.external_id == x_user_id).first()
            if app_user:
                resolved_email = app_user.email

        if not resolved_email:
            raise HTTPException(status_code=400, detail="Email required")

        query = db.query(Alert).filter(Alert.user_email == resolved_email)
        if not include_inactive:
            query = query.filter(Alert.is_active == True)  # noqa

        alerts = query.order_by(Alert.created_at.desc()).all()
        result: List[AlertOut] = []

        for a in alerts:
            best_run = (
                db.query(AlertRun)
                .filter(AlertRun.alert_id == a.id)
                .filter(AlertRun.price_found.isnot(None))
                .order_by(AlertRun.price_found.asc())
                .first()
            )

            best_price = best_run.price_found if best_run else None

            result.append(
                AlertOut(
                    id=a.id,
                    email=a.user_email,
                    origin=a.origin,
                    destination=a.destination,
                    cabin=a.cabin,
                    search_mode=a.search_mode,
                    departure_start=a.departure_start,
                    departure_end=a.departure_end,
                    return_start=a.return_start,
                    return_end=a.return_end,
                    alert_type=a.alert_type,
                    max_price=a.max_price,
                    mode=a.mode,
                    passengers=_derive_alert_passengers(a),
                    times_sent=a.times_sent,
                    is_active=a.is_active,
                    last_price=a.last_price,
                    best_price=best_price,
                    last_run_at=a.last_run_at,
                    last_notified_at=getattr(a, "last_notified_at", None),
                    last_notified_price=getattr(a, "last_notified_price", None),
                    created_at=a.created_at,
                    updated_at=a.updated_at,
                )
            )

        return result
    finally:
        db.close()


@app.patch("/alerts/{alert_id}")
def update_alert(
    alert_id: str,
    payload: AlertUpdatePayload,
    email: Optional[str] = None,
    x_user_id: Optional[str] = Header(None, alias="X-User-Id"),
):
    db = SessionLocal()
    try:
        resolved_email = email
        if not resolved_email and x_user_id:
            app_user = db.query(AppUser).filter(AppUser.external_id == x_user_id).first()
            if app_user:
                resolved_email = app_user.email

        if not resolved_email:
            raise HTTPException(status_code=400, detail="Email required")

        alert = (
            db.query(Alert)
            .filter(Alert.id == alert_id, Alert.user_email == resolved_email)
            .first()
        )
        if not alert:
            raise HTTPException(status_code=404, detail="Alert not found")

        for field in (
            "alert_type",
            "max_price",
            "departure_start",
            "departure_end",
            "return_start",
            "return_end",
            "mode",
            "is_active",
        ):
            val = getattr(payload, field, None)
            if val is not None:
                setattr(alert, field, val)

        if payload.passengers:
            max_passengers = get_config_int("MAX_PASSENGERS", 4)
            alert.passengers = min(max(1, payload.passengers), max_passengers)

        alert.updated_at = datetime.utcnow()
        db.commit()
        db.refresh(alert)

        return {"status": "ok"}
    finally:
        db.close()


@app.delete("/alerts/{alert_id}")
def delete_alert(
    alert_id: str,
    email: Optional[str] = None,
    x_user_id: Optional[str] = Header(None, alias="X-User-Id"),
):
    db = SessionLocal()
    try:
        alert = db.query(Alert).filter(Alert.id == alert_id).first()
        if not alert:
            raise HTTPException(status_code=404, detail="Alert not found")

        resolved_email = email
        if not resolved_email and x_user_id:
            app_user = db.query(AppUser).filter(AppUser.external_id == x_user_id).first()
            if app_user:
                resolved_email = app_user.email

        if resolved_email and alert.user_email != resolved_email:
            raise HTTPException(status_code=403, detail="Forbidden")

        db.query(AlertRun).filter(AlertRun.alert_id == alert.id).delete()
        db.delete(alert)
        db.commit()

        return {"status": "ok", "id": alert_id}
    finally:
        db.close()

# =====================================================================
# SECTION END: ALERT ROUTES
# =====================================================================
