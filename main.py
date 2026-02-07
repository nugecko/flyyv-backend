# =====================================================================
# SECTION START: IMPORTS
# =====================================================================

import os
from datetime import date, datetime, timedelta
from enum import Enum
from typing import List, Optional, Tuple, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from uuid import uuid4
from uuid import UUID
from sqlalchemy import text
from collections import defaultdict, Counter
import smtplib
from email.message import EmailMessage
from sqlalchemy import func
from datetime import datetime
from typing import Optional
from concurrent.futures import wait, FIRST_COMPLETED

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

# =====================================================================
# Alerts, v1 API models (used by /alerts routes)
# =====================================================================

class AlertCreate(BaseModel):
    # Legacy field, not trusted for auth or ownership, kept for backward compatibility
    email: Optional[str] = None

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

    # Still returned for UI display for now, ownership is via X-User-Id and alerts.user_external_id
    email: Optional[str] = None

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

    # Computed UI state
    state: str  # "active" | "paused" | "expired"

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

class TTNBookRequest(BaseModel):
    session_id: str
    recommendation_id: str
    email: Optional[str] = None

class TTNPassenger(BaseModel):
    type: str  # ADT, CHD, INF
    firstname: str
    lastname: str
    birthday: str  # DD-MM-YYYY
    gender: str  # M, F
    citizenship: str  # 2-letter
    docnum: str
    doc_expire_date: str  # DD-MM-YYYY

class TTNCheckoutLinkRequest(BaseModel):
    providerSessionId: str
    providerRecommendationId: str
    real_email: str
    real_phone: str
    passengers: List[TTNPassenger]

class SearchParams(BaseModel):
    origin: str
    destination: str
    user_external_id: Optional[str] = None
    earliestDeparture: date
    latestDeparture: date
    minStayDays: int
    maxStayDays: int
    nights: Optional[int] = None
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
    CANCELLED = "cancelled"

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

    # Provider identity, used for checkout/booking handoff
    provider: Optional[str] = None
    providerSessionId: Optional[str] = None
    providerRecommendationId: Optional[str] = None

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
    elapsedSeconds: Optional[float] = None
    estimatedTotalSeconds: Optional[float] = None
    estimatedProgressPct: Optional[float] = None


class SearchResultsResponse(BaseModel):
    jobId: str
    status: JobStatus
    totalResults: int
    offset: int
    limit: int
    options: List[FlightOption]

class UserSyncPayload(BaseModel):
    # Identity: accept any of these, canonicalised in /user-sync
    external_id: Optional[str] = None
    id: Optional[str] = None
    user_id: Optional[str] = None

    email: Optional[str] = None
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    country: Optional[str] = None
    marketing_consent: Optional[bool] = None
    source: Optional[str] = None

    # Base44 source-of-truth tier, expected values:
    # free, gold, platinum, tester, admin
    plan_tier_code: Optional[str] = None

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
    email: Optional[str] = None
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
    # Basic status
    is_active: Optional[bool] = None

    # Core alert rule
    alert_type: Optional[str] = None          # e.g. "scheduled_3x", "new_best", "under_price"
    mode: Optional[str] = None                # e.g. "smart" or "single"
    max_price: Optional[int] = None

    # Date windows (used for flex alerts and fixed alerts)
    departure_start: Optional[date] = None
    departure_end: Optional[date] = None
    return_start: Optional[date] = None
    return_end: Optional[date] = None

    # Passengers
    passengers: Optional[int] = None

    # Legacy or UI fields (safe to accept even if not used)
    preferred_days: Optional[List[int]] = None
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


class ProfileEntitlements(BaseModel):
    plan_tier: str
    active_alert_limit: int
    max_departure_window_days: int
    checks_per_day: int

class ProfileAlertUsage(BaseModel):
    active_alerts: int
    remaining_slots: int

class ProfileResponse(BaseModel):
    # Single source of truth fields for Base44 PlanCard
    display_name: str
    external_id: str
    joined_at: Optional[datetime] = None

    user: ProfileUser
    subscription: SubscriptionInfo
    wallet: WalletInfo
    entitlements: Optional[ProfileEntitlements] = None
    alertUsage: Optional[ProfileAlertUsage] = None

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

try:
    print(f"[guardrail] request_keys={sorted(list(params.model_dump().keys()))}")
except Exception:
    pass

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

def parse_iso8601_duration(duration_str: Optional[str]) -> Optional[int]:
    """Parse ISO 8601 duration string (e.g., 'PT9H30M') to minutes."""
    if not duration_str:
        return None
    import re
    match = re.match(r'^PT(?:(\d+)H)?(?:(\d+)M)?$', duration_str)
    if not match:
        return None
    hours = int(match.group(1) or 0)
    minutes = int(match.group(2) or 0)
    return hours * 60 + minutes

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

        # Debug log to stdout (Dokku logs)
    try:
        request_id = (
            resp.headers.get("Request-Id")
            or resp.headers.get("Duffel-Request-Id")
            or resp.headers.get("X-Request-Id")
            or resp.headers.get("X-Correlation-Id")
        )

        safe_body = (resp.text or "")
        safe_body = safe_body.replace("\n", "\\n").replace("\r", "\\r")

        if resp.status_code >= 400:
            print(f"Duffel POST {path} status={resp.status_code} request_id={request_id} body={safe_body[:2000]}")
        else:
            print(f"Duffel POST {path} status={resp.status_code} request_id={request_id}")
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

    # Debug log to stdout (Dokku logs)
    try:
        request_id = (
            resp.headers.get("Request-Id")
            or resp.headers.get("Duffel-Request-Id")
            or resp.headers.get("X-Request-Id")
            or resp.headers.get("X-Correlation-Id")
        )

        if resp.status_code >= 400:
            safe_body = (resp.text or "")
            safe_body = safe_body.replace("\n", "\\n").replace("\r", "\\r")
            print(
                f"Duffel GET {path} status={resp.status_code} request_id={request_id} body={safe_body[:1200]}"
            )
        else:
            print(f"Duffel GET {path} status={resp.status_code} request_id={request_id}")
    except Exception:
        pass


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

    # duffel_get() returns the "data" payload already
    if isinstance(res, list):
        return res
    if isinstance(res, dict) and "data" in res and isinstance(res["data"], list):
        return res["data"]
    return []

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
    
    # [duffel] offer_request summary log (micro-step)
    try:
        data = (resp_json or {}).get("data") or {}
        warnings = (resp_json or {}).get("warnings") or data.get("warnings") or []
        errors = (resp_json or {}).get("errors") or data.get("errors") or []

        offer_request_id = data.get("id") or data.get("offer_request_id") or "UNKNOWN"
        cabin_ack = data.get("cabin_class") or ((data.get("cabin") or {}).get("cabin_class")) or "UNKNOWN"

        # Duffel sometimes includes offers inline, often it does not. We count safely either way.
        inline_offers = data.get("offers") or []
        inline_offers_count = len(inline_offers) if isinstance(inline_offers, list) else 0

        slices = data.get("slices") or []
        if slices and isinstance(slices, list):
            s0 = slices[0] or {}
            o = s0.get("origin") or {}
            d = s0.get("destination") or {}
            origin_type = o.get("type") or "UNKNOWN"
            dest_type = d.get("type") or "UNKNOWN"
        else:
            origin_type = "UNKNOWN"
            dest_type = "UNKNOWN"

        print(
            f'[duffel] offer_request id={offer_request_id} '
            f'inline_offers={inline_offers_count} '
            f'cabin_ack={cabin_ack} '
            f'origin_type={origin_type} destination_type={dest_type} '
            f'warnings={len(warnings)} errors={len(errors)}'
        )
    except Exception as e:
        print(f"[duffel] offer_request summary log failed: {e}")

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
    print(f"[PRICE_DEBUG] total_amount={total_price}, pax={pax}, per_person={price}")
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

            duffel_duration = seg.get("duration")  # e.g., "PT9H30M"
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
    Build (departure_date, return_date) pairs.

    Rules:
      - Single searches: allow minStayDays..maxStayDays (existing behaviour)
      - FlyyvFlex (search_mode == "flexible"): trip length is fixed
        Use a single "nights" value (source of truth):
          1) params.nights if present
          2) else params.minStayDays
          3) else 0
        Generate only one return per departure:
          return = departure + nights
        Skip if return > latestDeparture
        processedPairs must be len(valid_pairs) downstream
    """
    earliest = getattr(params, "earliestDeparture", None)
    latest = getattr(params, "latestDeparture", None)

    if not earliest or not latest or earliest > latest:
        return []

    print(f"[pairs] search_mode={getattr(params,'search_mode',None)} earliest={earliest} latest={latest} nights={getattr(params,'nights',None)} minStayDays={getattr(params,'minStayDays',None)} maxStayDays={getattr(params,'maxStayDays',None)} maxDatePairs={getattr(params,'maxDatePairs',None)}")

    # Respect any param cap if it exists
    param_cap = getattr(params, "maxDatePairs", None)
    if isinstance(param_cap, int) and param_cap > 0:
        max_pairs = min(max_pairs, param_cap)

    pairs = []

    search_mode = getattr(params, "search_mode", None)

    # -----------------------------------------------------------------
    # FlyyvFlex: fixed nights, one pair per departure, skip out-of-window
    # -----------------------------------------------------------------
    if (search_mode == "flexible") or (getattr(params, "nights", None) is not None) or (
        (getattr(params, "minStayDays", None) is not None)
        and (getattr(params, "maxStayDays", None) is not None)
        and int(getattr(params, "minStayDays") or 0) > 0
        and int(getattr(params, "minStayDays") or 0) == int(getattr(params, "maxStayDays") or 0)
    ):

        nights = getattr(params, "nights", None)
        if nights is None:
            nights = getattr(params, "minStayDays", None)
        nights = int(nights or 0)
        if nights < 0:
            nights = 0

        # In FlyyvFlex, latestDeparture is the last allowed departure date.
        # Therefore, the last valid departure is latest - nights (so return stays aligned to fixed trip length).
        last_dep = latest - timedelta(days=nights)

        dep = earliest
        while dep <= last_dep and len(pairs) < max_pairs:
            ret = dep + timedelta(days=nights)
            pairs.append((dep, ret))
            dep = dep + timedelta(days=1)

        print(
            f"[pairs_flex] nights={nights} earliest={earliest} latest={latest} "
            f"last_dep={last_dep} len={len(pairs)} "
            f"first_pair={pairs[0] if pairs else None} "
            f"last_pair={pairs[-1] if pairs else None}"
        )

        return pairs

    # -----------------------------------------------------------------
    # Default: original behaviour (range of stay lengths)
    # -----------------------------------------------------------------
    min_stay = max(0, int(getattr(params, "minStayDays", 0) or 0))
    max_stay = max(min_stay, int(getattr(params, "maxStayDays", min_stay) or min_stay))

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

    # Micro-step: force city origins to airport IATA for Duffel
    for s in slices:
        o = s.get("origin")

        # Most calls pass strings ("LON", "TLV"), only transform when it's a city object.
        if isinstance(o, dict) and o.get("type") == "city":
            airports = o.get("airports") or []
            if airports:
                preferred = next((a for a in airports if a.get("iata_code") == "LHR"), None)
                chosen = preferred or airports[0]
                s["origin"] = {"iata_code": chosen.get("iata_code")}
                s["origin_type"] = "airport"

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

    offer_request_id = data.get("id")
    if not offer_request_id:
        print("[direct_only] no offer_request_id returned")
        return []

    try:
        offers_json = duffel_list_offers(offer_request_id, limit=per_pair_limit)

        # Duffel can be eventually consistent, retry once after a short delay if empty
        if not offers_json:
            import time
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


def run_duffel_scan(params: SearchParams) -> List[FlightOption]:
    print(f"[search] run_duffel_scan START origin={params.origin} dest={params.destination}")

    # -----------------------------
    # Caps and date pair generation
    # -----------------------------
    max_pairs, max_offers_pair, max_offers_total = effective_caps(params)
    print(f"[search] caps max_pairs={max_pairs} max_offers_pair={max_offers_pair} max_offers_total={max_offers_total}")

    date_pairs = generate_date_pairs(params, max_pairs=max_pairs)
    print(f"[search] generated {len(date_pairs)} date pairs")

    max_date_pairs = get_config_int("MAX_DATE_PAIRS_PER_ALERT", 40)
    if max_date_pairs and len(date_pairs) > max_date_pairs:
        print(f"[search] capping date_pairs from {len(date_pairs)} to {max_date_pairs} using MAX_DATE_PAIRS_PER_ALERT")
        date_pairs = date_pairs[:max_date_pairs]

    # Definitive truth for UI counters
    earliest = getattr(params, "earliestDeparture", None)
    latest = getattr(params, "latestDeparture", None)
    nights = getattr(params, "nights", None)
    pairs_preview = [(d.isoformat(), r.isoformat()) for d, r in date_pairs[:12]]
    print(
        "[pairs_final]"
        f" origin={params.origin}"
        f" dest={params.destination}"
        f" earliestDeparture={earliest}"
        f" latestDeparture={latest}"
        f" nights={nights}"
        f" totalPairs={len(date_pairs)}"
        f" preview12={pairs_preview}"
    )

    if not date_pairs:
        print("[search] no date pairs generated, returning empty list")
        return []

    # -----------------------------
    # Mitigation knobs
    # -----------------------------
    alert_scan_max_seconds = get_config_int("ALERT_SCAN_MAX_SECONDS", 180)  # hard stop, per alert run
    empty_pairs_stop_after = get_config_int("ALERT_EMPTY_PAIRS_STOP_AFTER", 12)  # consecutive empty pairs
    early_exit_min_results = get_config_int("ALERT_EARLY_EXIT_MIN_RESULTS", 200)  # stop if we already have enough
    early_exit_no_improve_pairs = get_config_int("ALERT_EARLY_EXIT_NO_IMPROVE_PAIRS", 10)  # pairs with no best improvement
    pair_cache_ttl_seconds = get_config_int("ALERT_PAIR_CACHE_TTL_SECONDS", 3600)  # reuse pair results across runs

    # -----------------------------
    # Pair ordering (try best first)
    # -----------------------------
    preferred_weekdays = {1: 0, 2: 1, 5: 2}  # Tue, Wed, Sat as "best first"; lower is better
    def pair_sort_key(dep_ret: Tuple[date, date]) -> Tuple[int, str]:
        dep, ret = dep_ret
        wd_rank = preferred_weekdays.get(dep.weekday(), 9)
        return (wd_rank, dep.isoformat())

    date_pairs = sorted(date_pairs, key=pair_sort_key)

    # -----------------------------
    # Simple in-process cache (PER SEARCH)
    # -----------------------------
    # IMPORTANT: do NOT use a global cache for raw Duffel offers,
    # it will grow across searches and can OOM the container.
    _DUFFEL_PAIR_CACHE: Dict[str, Dict[str, Any]] = {}  # key -> {"ts": float, "offers": List[dict]}

    def _cache_key(dep: date, ret: date) -> str:
        cabin = getattr(params, "cabin", None)
        return f"{params.origin}|{params.destination}|{dep.isoformat()}|{ret.isoformat()}|{cabin}|pax={params.passengers}"

    # -----------------------------
    # Main scan loop with early exits
    # -----------------------------
    import time
    start_ts = time.time()

    all_results: List[FlightOption] = []
    total_added = 0
    hit_global_cap = False
    empty_streak = 0

    best_price_seen: Optional[float] = None
    pairs_since_best_improve = 0

    for idx, (dep, ret) in enumerate(date_pairs, start=1):
        elapsed = time.time() - start_ts
        if alert_scan_max_seconds and elapsed >= alert_scan_max_seconds:
            print(f"[search] timeout reached after {int(elapsed)}s, stopping scan early, pairs_done={idx - 1}")
            break

        if total_added >= max_offers_total:
            print(f"[search] total_added {total_added} reached max_offers_total {max_offers_total}, stopping")
            hit_global_cap = True
            break

        if empty_pairs_stop_after and empty_streak >= empty_pairs_stop_after:
            print(f"[search] empty_streak {empty_streak} reached stop_after {empty_pairs_stop_after}, stopping scan early")
            break

        if early_exit_min_results and total_added >= early_exit_min_results and pairs_since_best_improve >= early_exit_no_improve_pairs:
            print(
                f"[search] early exit: total_added={total_added} >= {early_exit_min_results} and "
                f"pairs_since_best_improve={pairs_since_best_improve} >= {early_exit_no_improve_pairs}"
            )
            break

        print(f"[search] pair {idx}/{len(date_pairs)} dep={dep} ret={ret} current_total={total_added} empty_streak={empty_streak}")

        # -----------------------------
        # Fetch offers (cache first)
        # -----------------------------
        offers_json: List[dict] = []
        ck = _cache_key(dep, ret)
        cached = _DUFFEL_PAIR_CACHE.get(ck)
        if cached and (time.time() - cached.get("ts", 0)) <= pair_cache_ttl_seconds:
            offers_json = list(cached.get("offers") or [])
            print(f"[search] cache_hit for dep={dep} ret={ret} offers={len(offers_json)}")
        else:
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
                    empty_streak += 1
                    pairs_since_best_improve += 1
                    continue

                per_pair_limit = min(max_offers_pair, max_offers_total - total_added)

                inline = offer_request.get("offers") or []
                if inline:
                    offers_json = inline[:per_pair_limit]
                    print(f"[search] Duffel inline offers dep={dep} ret={ret} count={len(inline)} used={len(offers_json)}")
                else:
                    offers_json = duffel_list_offers(offer_request_id, limit=per_pair_limit)
                    print(f"[search] Duffel listed offers dep={dep} ret={ret} used={len(offers_json)} request_id={offer_request_id}")

                _DUFFEL_PAIR_CACHE[ck] = {"ts": time.time(), "offers": offers_json}
            except HTTPException as e:
                print(f"[search] Duffel HTTPException dep={dep} ret={ret}: {e.detail}")
                empty_streak += 1
                pairs_since_best_improve += 1
                continue
            except Exception as e:
                print(f"[search] Unexpected Duffel error dep={dep} ret={ret}: {e}")
                empty_streak += 1
                pairs_since_best_improve += 1
                continue

        if not offers_json:
            print(f"[search] no offers to map for dep={dep} ret={ret}")
            empty_streak += 1
            pairs_since_best_improve += 1
            continue

        # -----------------------------
        # Map, filter, balance, add
        # -----------------------------
        empty_streak = 0

        mapped_pair: List[FlightOption] = [
            map_duffel_offer_to_option(offer, dep, ret, passengers=params.passengers)
            for offer in offers_json
        ]
        print(f"[search] pair dep={dep} ret={ret}: mapped {len(mapped_pair)} offers")

        filtered_pair = apply_filters(mapped_pair, params)
        print(f"[search] pair dep={dep} ret={ret}: filtered down to {len(filtered_pair)} offers")

        if not filtered_pair:
            print(f"[search] pair dep={dep} ret={ret}: no offers after filters")
            pairs_since_best_improve += 1
            continue

        filtered_pair.sort(key=lambda o: (getattr(o, "price", float("inf"))))
        pair_best = float(getattr(filtered_pair[0], "price", float("inf")))
        if best_price_seen is None or pair_best < best_price_seen:
            best_price_seen = pair_best
            pairs_since_best_improve = 0
        else:
            pairs_since_best_improve += 1

        airline_counts_pair = Counter(opt.airlineCode or opt.airline for opt in filtered_pair)
        print(f"[search] airline mix before balance for dep={dep} ret={ret}: {dict(airline_counts_pair)}")

        if len(filtered_pair) > max_offers_pair:
            print(
                f"[search] pair dep={dep} ret={ret}: capping offers from {len(filtered_pair)} to {max_offers_pair} using max_offers_pair"
            )
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

    elapsed_total = int(time.time() - start_ts)
    print(
        f"[search] run_duffel_scan DONE, returning {len(all_results)} offers "
        f"from {len(date_pairs)} date pairs, hit_global_cap={hit_global_cap}, elapsed_seconds={elapsed_total}"
    )
    return all_results

# =====================================================================
# SECTION END: SHARED SEARCH HELPERS
# =====================================================================

# ============================================================
# ASYNC DATE-PAIR WORKER (CRITICAL)
# ============================================================
# This function is intentionally small and boring.
#
# WHY IT EXISTS:
# - Async searches fan out into multiple date-pairs
# - Each date-pair must behave exactly like a normal
#   single-pair Duffel search
# - We MUST NOT share SearchParams across threads
#
# DO NOT:
# - Call Duffel directly here
# - Re-implement filtering, curation, or airline caps
# - Mutate the original params object
#
# WHAT THIS FUNCTION DOES:
# 1. Clone SearchParams for a single dep/ret pair
# 2. Force the search into single-pair mode
# 3. Call the existing sync pipeline (run_duffel_scan)
# 4. Return raw FlightOption results for this pair
#
# WHY THIS DESIGN:
# - Keeps async and sync logic identical
# - Prevents thread safety bugs
# - Allows cancellation by killing the executor
#
# If async jobs hang or behave differently from sync,
# this function is the FIRST place to check.
# ============================================================
def process_date_pair_offers(
    params: SearchParams,
    dep: date,
    ret: date,
    max_offers_pair: int,
) -> List[FlightOption]:
    """
    Fetch offers for exactly one (dep, ret) pair using TTN, then map to FlightOption.
    TTN scan is executed per departure date (probe-first), then we stamp dep/ret onto results.
    """

    per_pair_limit = int(max_offers_pair) if max_offers_pair else 20
    per_pair_limit = max(1, min(per_pair_limit, 50))

    try:
        # Create a copy of params and force the single departure date for this pair.
        # We keep destination/origin/cabin/passengers from the incoming params.
        scan_params = SearchParams(**params.model_dump())
        scan_params.earliestDeparture = dep

        opts = run_ttn_scan(scan_params) or []

        # Stamp the pair dates onto the returned options so Flyyv UI logic stays consistent
        dep_iso = dep.isoformat()
        ret_iso = ret.isoformat()

        for o in opts:
            try:
                o.departureDate = dep_iso
                o.returnDate = ret_iso
            except Exception:
                pass

        # Keep per-pair cap behavior consistent with the rest of the pipeline
        opts = opts[:per_pair_limit]

        print(f"[pair_worker] TTN dep={dep_iso} ret={ret_iso} mapped={len(opts)}")
        return opts

    except HTTPException as e:
        print(f"[pair_worker] TTN HTTPException dep={dep} ret={ret}: {e.detail}")
        return []
    except Exception as e:
        print(f"[pair_worker] TTN error dep={dep} ret={ret}: {e}")
        return []

# ============================================================
# END - ASYNC DATE-PAIR WORKER (CRITICAL)
# ============================================================

# ============================================================
# TTN API HELPERS (probe-only)
# Notes:
# - TTN expects query parameter named "key" for auth.
# - /avia/search is GET-only, use /avia/search.json to get JSON.
# - For now: probe mode plus minimal mapping, returns up to 3 FlightOptions.
# ============================================================

TTN_BASE_URL = "https://v2.api.tickets.ua"

def _get_ttn_auth_key() -> Optional[str]:
    return (os.getenv("TTN_AUTH_KEY") or "").strip() or None

def _get_ttn_api_key() -> Optional[str]:
    # Prefer env var for secrets, fallback to admin_config if you later store it there
    return os.getenv("TTN_API_KEY") or get_config_str("TTN_API_KEY", None)


def _ttn_headers() -> Dict[str, str]:
    # Keep headers minimal, TTN auth is passed via query param "key"
    return {
        "Accept": "application/json",
        "Content-Type": "application/json",
    }


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
    # Not used for /avia/search (GET-only), but kept for later endpoints
    api_key = _get_ttn_api_key()
    if not api_key:
        raise HTTPException(status_code=500, detail="TTN_API_KEY is not configured")

    url = f"{TTN_BASE_URL}{path}"

    merged_params = dict(params or {})
    merged_params.setdefault("key", api_key)

    print(f"[ttn] POST {path} params={merged_params} payload_keys={list((payload or {}).keys())}")

    res = requests.post(
        url,
        params=merged_params,
        json=payload,
        headers=_ttn_headers(),
        timeout=30,
    )

    if res.status_code >= 400:
        raise HTTPException(
            status_code=502,
            detail=f"TTN POST {path} failed: {res.status_code} {res.text}",
        )

    return res.json()

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
    import hashlib
    import json as _json

    if not isinstance(offer, dict):
        raise ValueError("TTN offer is not a dict")

    # -------------------------
    # Price
    # -------------------------
    cur = offer.get("currency") or "EUR"

    amt = offer.get("amount")
    amt_val = None
    if isinstance(amt, dict) and cur:
        amt_val = amt.get(cur)
    elif amt is not None:
        amt_val = amt

    if amt_val is None:
        fare = offer.get("fare")
        taxes = offer.get("taxes")
        if fare is not None and taxes is not None:
            try:
                amt_val = float(fare) + float(taxes)
            except Exception:
                amt_val = None

    price = float(amt_val) if amt_val is not None else 0.0

    # -------------------------
    # IDs
    # -------------------------
    raw_id = offer.get("id") or offer.get("recommendation_id") or offer.get("rec_id") or offer.get("uid")
    if raw_id:
        opt_id = f"ttn:{raw_id}"
    else:
        payload_str = _json.dumps(offer, sort_keys=True, default=str)
        h = hashlib.md5(payload_str.encode("utf-8")).hexdigest()[:16]
        opt_id = f"ttn:hash:{h}"

    # -------------------------
    # Helpers
    # -------------------------
    def _clean_dt(s):
        if not s:
            return None
        s = str(s).strip()
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        return s

    def _try_parse_dt(s):
        s2 = _clean_dt(s)
        if not s2:
            return None
        try:
            return datetime.fromisoformat(s2)
        except Exception:
            for fmt in (
                "%d-%m-%Y %H:%M",
                "%d-%m-%Y %H:%M:%S",
                "%d-%m-%Y",
                "%d.%m.%Y %H:%M",   # TTN e.g. 01.03.2026 06:20
                "%d.%m.%Y %H:%M:%S",
                "%d.%m.%Y",
            ):
                try:
                    return datetime.strptime(s2, fmt)
                except Exception:
                    continue
        return None

    def _iata_from(seg, keys):
        for k in keys:
            v = seg.get(k)
            if isinstance(v, str) and v.strip():
                return v.strip().upper()
            if isinstance(v, dict):
                for kk in ("iata", "iata_code", "code"):
                    vv = v.get(kk)
                    if isinstance(vv, str) and vv.strip():
                        return vv.strip().upper()
        return None

    def _name_from(seg, keys):
        for k in keys:
            v = seg.get(k)
            if isinstance(v, str) and v.strip():
                return v.strip()
            if isinstance(v, dict):
                for kk in ("name", "title"):
                    vv = v.get(kk)
                    if isinstance(vv, str) and vv.strip():
                        return vv.strip()
        return None

    def _carrier_code(seg):
        # Common TTN and generic keys for airline IATA code
        for k in (
            "marketingCarrier",
            "marketing_carrier",
            "carrier",
            "airline",
            "operatingCarrier",
            "operating_carrier",
            "airline_code",
            "carrier_code",
            "iata",
            "iata_code",
            "airline_iata",
            "operating_airline",
            "marketing_airline",
        ):
            v = seg.get(k)

            if isinstance(v, str) and v.strip():
                vv = v.strip().upper()
                # Guard against values like "TK,TK"
                if "," in vv:
                    vv = vv.split(",")[0].strip()
                return vv

            if isinstance(v, dict):
                for kk in ("iata", "iata_code", "code"):
                    vv = v.get(kk)
                    if isinstance(vv, str) and vv.strip():
                        return vv.strip().upper()

        return None

    def _flight_number(seg):
        for k in ("flightNumber", "flight_number", "marketing_carrier_flight_number", "number"):
            v = seg.get(k)
            if v is None:
                continue
            v = str(v).strip()
            if v:
                return v
        return None

    # -------------------------
    # Build segments
    # -------------------------
    stopover_codes = []
    stopover_airports = []
    aircraft_codes = []
    aircraft_names = []

    def _build_segments(direction, segs):
        seg_out = []
        total_min = 0
        first_dep = None
        last_arr = None

        for i, s in enumerate(segs):
            if not isinstance(s, dict):
                continue

            dep_dt = (
                _try_parse_dt(s.get("departingAt"))
                or _try_parse_dt(s.get("departure_datetime"))
                or _try_parse_dt(s.get("departure_at"))
                or _try_parse_dt(s.get("departure"))
                or _try_parse_dt(s.get("dep_time"))
                or _try_parse_dt(s.get("departure_time"))  # TTN
            )
            arr_dt = (
                _try_parse_dt(s.get("arrivingAt"))
                or _try_parse_dt(s.get("arrival_datetime"))
                or _try_parse_dt(s.get("arrival_at"))
                or _try_parse_dt(s.get("arrival"))
                or _try_parse_dt(s.get("arr_time"))
                or _try_parse_dt(s.get("arrival_time"))  # TTN
            )
            
            o_code = _iata_from(
                s,
                ["origin", "from", "departure_airport", "departureAirport", "dep_airport", "dep", "departure_airport"],
            )
            d_code = _iata_from(
                s,
                ["destination", "to", "arrival_airport", "arrivalAirport", "arr_airport", "arr", "arrival_airport"],
            )

            # TTN provides airport names separately
            o_name = _name_from(s, ["originAirport", "departure_airport_name", "departureAirport", "origin", "departure_airport"])
            d_name = _name_from(s, ["destinationAirport", "arrival_airport_name", "arrivalAirport", "destination", "arrival_airport"])

            mkt = _carrier_code(s)
            fn = _flight_number(s)

            seg_min = 0
            if dep_dt and arr_dt:
                try:
                    seg_min = int((arr_dt - dep_dt).total_seconds() // 60)
                    if seg_min < 0:
                        seg_min = 0
                except Exception:
                    seg_min = 0
            else:
                dur = s.get("durationMinutes") or s.get("duration") or s.get("segment_duration")
                try:
                    seg_min = int(dur) if dur is not None else 0
                except Exception:
                    seg_min = 0

            ac_code = None
            ac_name = None
            if isinstance(s.get("aircraft"), dict):
                ac_code = s["aircraft"].get("code") or s["aircraft"].get("iata_code")
                ac_name = s["aircraft"].get("name")
            else:
                ac_code = s.get("aircraftCode") or s.get("aircraft_code")
                ac_name = s.get("aircraftName") or s.get("aircraft_name")

            if ac_code:
                aircraft_codes.append(str(ac_code))
            if ac_name:
                aircraft_names.append(str(ac_name))

            dep_str = dep_dt.isoformat() if dep_dt else _clean_dt(s.get("departingAt")) or _clean_dt(s.get("departure_datetime"))
            arr_str = arr_dt.isoformat() if arr_dt else _clean_dt(s.get("arrivingAt")) or _clean_dt(s.get("arrival_datetime"))

            seg_out.append(
                {
                    "direction": direction,
                    "flightNumber": fn,
                    "marketingCarrier": mkt,
                    "operatingCarrier": None,
                    "origin": o_code,
                    "destination": d_code,
                    "originAirport": o_name,
                    "destinationAirport": d_name,
                    "departingAt": dep_str,
                    "arrivingAt": arr_str,
                    "aircraftCode": ac_code,
                    "aircraftName": ac_name,
                    "durationMinutes": seg_min,
                    "layoverMinutesToNext": None,
                    "cabin": None,
                    "bookingCode": None,
                    "fareBrand": None,
                }
            )

            total_min += seg_min

            if dep_dt and first_dep is None:
                first_dep = dep_dt
            if arr_dt:
                last_arr = arr_dt

            if i < len(segs) - 1:
                if d_code:
                    stopover_codes.append(d_code)
                if d_name:
                    stopover_airports.append(d_name)

        return seg_out, total_min, first_dep, last_arr

    # -------------------------
    # Routes
    # -------------------------
    routes = offer.get("routes")
    outbound_segments = []
    return_segments = []
    stops = 0
    total_duration_minutes = 0

    dep_date_str = dep_date.isoformat()
    ret_date_str = ret_date.isoformat()

    airline_code = None

    if isinstance(routes, list) and routes:
        # outbound
        r_out = routes[0] if isinstance(routes[0], dict) else None
        out_min = 0
        if r_out:
            segs = r_out.get("segments")
            if isinstance(segs, list) and segs:
                outbound_segments, out_min, out_first, out_last = _build_segments("outbound", segs)
                total_duration_minutes += out_min
                stops = max(stops, max(0, len(segs) - 1))
                if out_first:
                    dep_date_str = out_first.date().isoformat()
                if outbound_segments and outbound_segments[0].get("marketingCarrier"):
                    airline_code = outbound_segments[0].get("marketingCarrier")

            # fallback route_duration if segment durations missing
            try:
                rd = int(r_out.get("route_duration") or 0)
                if rd > 0 and out_min == 0:
                    total_duration_minutes += rd
            except Exception:
                pass

        # inbound
        in_min = 0
        if len(routes) > 1:
            r_in = routes[1] if isinstance(routes[1], dict) else None
            if r_in:
                segs = r_in.get("segments")
                if isinstance(segs, list) and segs:
                    return_segments, in_min, in_first, in_last = _build_segments("return", segs)
                    total_duration_minutes += in_min
                    stops = max(stops, max(0, len(segs) - 1))
                    if in_first:
                        ret_date_str = in_first.date().isoformat()
                    elif in_last:
                        ret_date_str = in_last.date().isoformat()
                try:
                    rd = int(r_in.get("route_duration") or 0)
                    if rd > 0 and in_min == 0:
                        total_duration_minutes += rd
                except Exception:
                    pass

    duration_minutes = int(total_duration_minutes) if total_duration_minutes else 0

    # Derive airline codes from segments, collapse duplicates, keep order
    def _unique_codes(seq):
        seen = set()
        out = []
        for x in seq:
            if not x:
                continue
            x = str(x).strip().upper()
            if "," in x:
                x = x.split(",")[0].strip()
            if x and x not in seen:
                seen.add(x)
                out.append(x)
        return out

    outbound_codes = _unique_codes([s.get("marketingCarrier") for s in (outbound_segments or [])])
    return_codes = _unique_codes([s.get("marketingCarrier") for s in (return_segments or [])])
    all_codes = _unique_codes(outbound_codes + return_codes)

    # Always set a stable airlineCode for filtering, prefer first outbound carrier
    airline_code = (outbound_codes[0] if outbound_codes else (all_codes[0] if all_codes else None))

    # Label: if single airline, use that code for now, else show combined
    if len(all_codes) == 1:
        airline_name = all_codes[0]
    elif len(all_codes) > 1:
        airline_name = " + ".join(all_codes[:2]) if len(all_codes) <= 2 else f"{all_codes[0]} + {len(all_codes)-1} more"
    else:
        airline_name = "TTN"

    return FlightOption(
        id=opt_id,
        provider="ttn",
        providerSessionId=str(session_id) if session_id else None,
        providerRecommendationId=str(raw_id) if raw_id else None,
        airline=str(airline_name),
        airlineCode=airline_code,
        price=price,
        currency=str(cur),
        departureDate=dep_date_str,
        returnDate=ret_date_str,
        stops=int(stops),
        cabinSummary=None,
        cabinHighest=None,
        cabinByDirection=None,
        durationMinutes=duration_minutes,
        totalDurationMinutes=duration_minutes if duration_minutes else None,
        duration=None,
        origin=str(origin),
        destination=str(destination),
        originAirport=str(origin),
        destinationAirport=str(destination),
        stopoverCodes=list(dict.fromkeys(stopover_codes)) if stopover_codes else None,
        stopoverAirports=list(dict.fromkeys(stopover_airports)) if stopover_airports else None,
        outboundSegments=outbound_segments or None,
        returnSegments=return_segments or None,
        aircraftCodes=aircraft_codes or None,
        aircraftNames=aircraft_names or None,
        bookingUrl=None,
        url=None,
    )

def run_ttn_scan(
    params: SearchParams,
    dep_override: Optional[date] = None,
    ret_override: Optional[date] = None,
) -> List[FlightOption]:
    print(f"[ttn] run_ttn_scan START origin={getattr(params,'origin',None)} dest={getattr(params,'destination',None)}")

    # Use override when called from date-pair worker, otherwise fall back to params
    dep = dep_override or getattr(params, "earliestDeparture", None) or getattr(params, "departure_date", None)

    if not dep or not getattr(params, "origin", None) or not getattr(params, "destination", None):
        print("[ttn] missing required params (origin/destination/dep), skipping TTN scan")
        return []

    # TTN expects DD-MM-YYYY
    if hasattr(dep, "strftime"):
        dep_str = dep.strftime("%d-%m-%Y")
    else:
        dep_str = str(dep)

    pax = int(getattr(params, "passengers", 1) or 1)

    # Map Flyyv cabin to TTN service_class: A=all, E=economy, B=business
    cabin_raw = (getattr(params, "cabin", None) or "BUSINESS").upper()
    if cabin_raw in ("BUSINESS", "B"):
        service_class = "B"
    elif cabin_raw in ("ECONOMY", "E"):
        service_class = "E"
    else:
        service_class = "A"

        # Compute return date if we can
    ret_date_obj = None

    # 1) Prefer explicit override from date-pair worker
    if isinstance(ret_override, date):
        ret_date_obj = ret_override

    # 2) Otherwise infer from fixed nights (FlyyvFlex style)
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

    qs = {
        "destinations[0][departure]": params.origin,
        "destinations[0][arrival]": params.destination,
        "destinations[0][date]": dep_str,
        "adt": pax,
        "service_class": service_class,
        "lang": "en",
    }

    # Roundtrip only when we have a return date
    if ret_str:
        qs.update(
            {
                "destinations[1][departure]": params.destination,
                "destinations[1][arrival]": params.origin,
                "destinations[1][date]": ret_str,
            }
        )

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

                # Small sample logs, only for the first 2 items
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

                        # 🔑 Checkout / booking candidates
                        cand_keys = []
                        for k in r0.keys():
                            kl = str(k).lower()
                            if any(s in kl for s in ["url", "link", "book", "pay", "payment", "order", "redirect", "checkout", "fat", "token"]):
                                cand_keys.append(k)

                        if cand_keys:
                            cand = {k: r0.get(k) for k in cand_keys}
                            print(
                                f"[ttn] rec.checkout_candidates[{sample_printed}] "
                                f"keys={cand_keys} values_preview={str(cand)[:800]}"
                            )
                        else:
                            print(f"[ttn] rec.checkout_candidates[{sample_printed}] keys=[]")

                        sample_printed += 1

                    cur = r0.get("currency")
                    amt = r0.get("amount")
                    if isinstance(amt, dict) and cur:
                        amt_val = amt.get(cur)
                    else:
                        amt_val = amt

                    if amt_val is None:
                        fare = r0.get("fare")
                        taxes = r0.get("taxes")
                        if fare is not None and taxes is not None:
                            try:
                                amt_val = float(fare) + float(taxes)
                            except Exception:
                                amt_val = None

                    if amt_val is None:
                        continue

                    try:
                        val = float(amt_val)
                    except Exception:
                        continue

                    if cheapest is None or val < cheapest:
                        cheapest = val
                        cheapest_currency = cur

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

        # ---- Map and return a small set of TTN options (preview mode) ----
    # Goal: always return a consistent list (possibly empty) without throwing.
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

        # Use override return date when provided, otherwise keep same-day default
                # Use the same return date logic as the TTN request
        ret_date_obj = None

        if isinstance(ret_override, date):
            ret_date_obj = ret_override
        else:
            try:
                nights = getattr(params, "minStayDays", None)
                if nights is None:
                    nights = getattr(params, "nights", None)
                if nights is not None:
                    nights = int(nights)
                    if nights > 0:
                        ret_date_obj = dep_date_obj + timedelta(days=nights)
            except Exception:
                ret_date_obj = None

        if ret_date_obj is None:
            ret_date_obj = dep_date_obj


        mapped: List[FlightOption] = []
        attempted = 0

        for r0 in recs:
            if len(mapped) >= 3:
                break
            attempted += 1

            if not isinstance(r0, dict):
                continue

            try:
                opt = map_ttn_offer_to_option(
                    r0,
                    dep_date=dep_date_obj,
                    ret_date=ret_date_obj,
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
    options = run_ttn_scan(params)

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


def _min_interval_seconds_for_checks_per_day(n: int) -> int:
    """
    Plan enforcement: scheduler frequency by plan_checks_per_day
    1/day => 24h cadence, 3/day => 8h cadence
    """
    try:
        n = int(n or 1)
    except Exception:
        n = 1

    if n <= 1:
        return 24 * 60 * 60
    if n >= 3:
        return 8 * 60 * 60
    return int((24 * 60 * 60) / n)


def _get_user_for_alert(db: Session, alert: Alert) -> Optional[AppUser]:
    """
    Identity lookup for cadence, entitlements, and email eligibility.

    Priority:
    1) AppUser.external_id == alert.user_external_id (preferred, matches X-User-Id)
    2) AppUser.email == alert.user_email (legacy fallback only)
    """
    user_external_id = getattr(alert, "user_external_id", None)
    if user_external_id:
        user = db.query(AppUser).filter(AppUser.external_id == user_external_id).first()
        if user:
            return user

    user_email = getattr(alert, "user_email", None)
    if user_email:
        return db.query(AppUser).filter(AppUser.email == user_email).first()

    return None


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
    # =====================================================================
    # SECTION: INIT AND START LOG
    # =====================================================================
    now = datetime.utcnow()

    print(
        f"[alerts] process_alert START "
        f"run_id=pending "
        f"id={alert.id} "
        f"external_id={getattr(alert, 'user_external_id', None)} "
        f"email={getattr(alert, 'user_email', None)} "
        f"type={getattr(alert, 'alert_type', None)} "
        f"mode={getattr(alert, 'mode', None)}"
    )

    # =====================================================================
    # SECTION: USER RESOLUTION AND GLOBAL ENABLEMENT
    # =====================================================================
    user = _get_user_for_alert(db, alert)

    # Always stamp last_run_at when a run is attempted, even if we exit early
    alert.last_run_at = now
    alert.updated_at = now

    # Create a run id up front so ALL early exits can write an AlertRun row safely
    alert_run_id = str(uuid4())

    if not user:
        db.add(
            AlertRun(
                id=alert_run_id,
                alert_id=alert.id,
                run_at=now,
                price_found=None,
                sent=False,
                reason="no_user_for_alert",
            )
        )
        db.commit()
        return

    if not should_send_alert(db, user):
        db.add(
            AlertRun(
                id=alert_run_id,
                alert_id=alert.id,
                run_at=now,
                price_found=None,
                sent=False,
                reason="alerts_disabled",
            )
        )
        db.commit()
        return

    # =====================================================================
    # SECTION: DUPLICATE GUARD (SHORT WINDOW)
    # =====================================================================
    if getattr(alert, "last_checked_at", None) is not None:
        age_seconds = (now - alert.last_checked_at).total_seconds()
        if age_seconds < 300:
            print(
                f"[alerts] skip recent_check "
                f"run_id={alert_run_id} alert_id={alert.id} age_seconds={int(age_seconds)}"
            )
            return

    # Stamp last_checked_at once, after passing the guard
    alert.last_checked_at = now
    alert.updated_at = now
    db.commit()

    # =======================================
    # SECTION: CREATE ALERT RUN ROW (FK ANCHOR)
    # Creates the alert_runs row up front so snapshots can FK to it safely.
    # =======================================

    run_row = AlertRun(
        id=alert_run_id,
        alert_id=str(alert.id),
        run_at=now,
        price_found=None,
        sent=False,
        reason="started",
    )
    db.add(run_row)
    db.commit()

    # =====================================================================
    # SECTION: BUILD SEARCH PARAMS AND SCAN PARAMS
    # =====================================================================
    params = build_search_params_for_alert(alert)

    # For under-price alerts, do not apply maxPrice during the scan itself.
    if getattr(alert, "max_price", None) is not None:
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

    # =====================================================================
    # SECTION: CACHE READ OR DUFFEL SCAN
    # =====================================================================
    import json
    from types import SimpleNamespace

    cache_hit = False
    options = None

    # Cache read, alert-only cache, reuse if still valid
    if alert.cache_expires_at and alert.cache_expires_at > now and alert.cache_payload_json:
        try:
            cached_list = json.loads(alert.cache_payload_json) or []
            options = [SimpleNamespace(**d) for d in cached_list]
            cache_hit = True
        except Exception as e:
            print(f"[alerts] cache read failed run_id={alert_run_id} alert_id={alert.id}: {e}")
            options = None
            cache_hit = False

    # Cache miss, run TTN
    if options is None:
        options = run_ttn_scan(scan_params)

        # Cache write, store even empty to avoid hammering Duffel for dead windows
        def _opt_to_dict(o):
            if hasattr(o, "model_dump"):
                return o.model_dump()
            if hasattr(o, "dict"):
                return o.dict()
            return dict(getattr(o, "__dict__", {}))

        try:
            alert.cache_created_at = now
            checks_per_day = max(1, user.plan_checks_per_day or 1)
            ttl_hours = 24 / checks_per_day
            alert.cache_expires_at = now + timedelta(hours=ttl_hours)
            alert.cache_payload_json = json.dumps([_opt_to_dict(o) for o in options])
            alert.updated_at = now
            db.commit()
        except Exception as e:
            print(f"[alerts] cache write failed run_id={alert_run_id} alert_id={alert.id}: {e}")

    print(
        f"[alerts] scan complete "
        f"run_id={alert_run_id} alert_id={alert.id} "
        f"options_count={len(options) if options else 0} "
        f"cache_hit={cache_hit} "
        f"scan_maxPrice={getattr(scan_params, 'maxPrice', None)} "
        f"plan_checks_per_day={getattr(user, 'plan_checks_per_day', None)}"
    )

    # =====================================================================
    # SECTION: NO RESULTS
    # =====================================================================
    if not options:
        db.add(
            AlertRun(
                id=alert_run_id,
                alert_id=alert.id,
                run_at=now,
                price_found=None,
                sent=False,
                reason="no_results_scan_empty",
            )
        )
        db.commit()
        return

    # =====================================================================
    # SECTION: CHEAPEST AND STORED BEST
    # =====================================================================
    options_sorted = sorted(options, key=lambda o: o.price)
    cheapest = options_sorted[0]
    current_price = int(cheapest.price)

    stored_best_price = getattr(alert, "last_price", None)
    if stored_best_price is None:
        best_run = (
            db.query(AlertRun)
            .filter(AlertRun.alert_id == alert.id)
            .filter(AlertRun.price_found.isnot(None))
            .order_by(AlertRun.price_found.asc())
            .first()
        )
        stored_best_price = int(best_run.price_found) if best_run and best_run.price_found is not None else None

    # =====================================================================
    # SECTION: DECIDE WHETHER TO SEND
    # =====================================================================
    should_send = False
    send_reason = None

    effective_type = getattr(alert, "alert_type", None) or "new_best"
    if effective_type == "price_change":
        effective_type = "new_best"
    elif effective_type == "scheduled_3x":
        effective_type = "summary"

    max_price_threshold = getattr(alert, "max_price", None)
    if max_price_threshold is not None:
        effective_type = "under_price"

    if effective_type == "under_price":
        if current_price <= int(max_price_threshold):
            should_send = True
            send_reason = "under_price"
        else:
            send_reason = "not_under_price"

    elif effective_type == "new_best":
        if stored_best_price is None:
            should_send = True
            send_reason = "first_best"
        elif current_price < int(stored_best_price):
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

    # =====================================================================
    # SECTION: SNAPSHOT INSERT (ONLY WHEN WE PLAN TO SEND)
    # =====================================================================
    if should_send:
        try:
            from sqlalchemy import text as sql_text

            def _opt_to_dict_for_snapshot(opt):
                if opt is None:
                    return {}
                if hasattr(opt, "model_dump"):
                    return opt.model_dump()
                if hasattr(opt, "dict"):
                    return opt.dict()
                data = getattr(opt, "__dict__", None)
                return dict(data) if isinstance(data, dict) else {}

            def _to_float(val):
                if val is None:
                    return None
                if isinstance(val, (int, float)):
                    return float(val)
                try:
                    return float(str(val).strip())
                except Exception:
                    return None

            def _pick_first(d, keys):
                for k in keys:
                    if k in d and d.get(k) is not None:
                        return d.get(k)
                return None

            # Store a small, stable payload, top 5 is enough to render the email and page consistently
            pax = max(1, int(getattr(params, "passengers", 1) or 1))
            raw_top_results = [_opt_to_dict_for_snapshot(o) for o in (options_sorted or [])[:5]]

            top_results = []
            for item in raw_top_results:
                if not isinstance(item, dict):
                    top_results.append(item)
                    continue

                # Prefer explicit fields if present, otherwise derive deterministically
                price_per_pax = _to_float(
                    _pick_first(item, ["price_per_pax", "pricePerPax", "per_pax_price", "price"])
                )
                total_price = _to_float(
                    _pick_first(item, ["total_price", "totalPrice", "total"])
                )

                if price_per_pax is None and total_price is not None:
                    price_per_pax = round(total_price / pax, 2)

                if total_price is None and price_per_pax is not None:
                    total_price = round(price_per_pax * pax, 2)

                item["passengers"] = pax
                item["price_per_pax"] = price_per_pax
                item["total_price"] = total_price

                top_results.append(item)

            # Params: best effort to persist inputs that explain the run
            params_payload = {}
            for k in [
                "origin",
                "destination",
                "cabin",
                "passengers",
                "search_mode",
                "earliestDeparture",
                "latestDeparture",
                "nights",
                "minStayDays",
                "maxStayDays",
                "stopsFilter",
                "maxPrice",
                "currency",
            ]:
                try:
                    v = getattr(params, k, None)
                    if hasattr(v, "isoformat"):
                        v = v.isoformat()
                    params_payload[k] = v
                except Exception:
                    pass

            currency = params_payload.get("currency") or "GBP"

            best_price_val = _to_float(current_price)
            best_price_int = int(round(best_price_val)) if best_price_val is not None else None

            db.execute(
                sql_text(
                    """
                    insert into alert_run_snapshots
                        (alert_run_id, alert_id, user_email, params, top_results, best_price_per_pax, currency, meta)
                    values
                        (:alert_run_id, :alert_id, :user_email,
                         CAST(:params AS jsonb), CAST(:top_results AS jsonb),
                         :best_price, :currency, CAST(:meta AS jsonb))
                    on conflict (alert_run_id) do nothing
                    """
                ),
                {
                    "alert_run_id": alert_run_id,
                    "alert_id": str(alert.id),
                    "user_email": getattr(alert, "user_email", None),
                    "params": json.dumps(params_payload),
                    "top_results": json.dumps(top_results),
                    "best_price": best_price_int,
                    "currency": str(currency),
                    "meta": json.dumps(
                        {
                            "cache_hit": bool(cache_hit),
                            "options_count": int(len(options_sorted or [])),
                            "top_results_count": int(len(top_results)),
                            "send_reason": str(send_reason),
                            "created_at_utc": now.isoformat(),
                        }
                    ),
                },
            )
            db.commit()

            print(
                f"[alerts] snapshot saved "
                f"run_id={alert_run_id} alert_id={alert.id} "
                f"best_price_per_pax={best_price_int} currency={currency}"
            )
        except Exception as e:
            # If snapshot fails, we still send, but we log loudly.
            print(f"[alerts] snapshot insert failed run_id={alert_run_id} alert_id={alert.id}: {e}")
            try:
                db.rollback()
            except Exception:
                pass

    # =====================================================================
    # SECTION: SEND EMAIL
    # =====================================================================
    
    sent_flag = False

    if should_send:
        try:
            if getattr(alert, "mode", None) == "smart":
                # NOTE: we will pass alert_run_id into the email builder in the next tiny edit.
                send_smart_alert_email(alert, options_sorted, params, alert_run_id=alert_run_id)
            else:
                send_alert_email_for_alert(alert, cheapest, params, alert_run_id=alert_run_id)
            sent_flag = True
        except Exception as e:
            print(f"[alerts] Failed to send email run_id={alert_run_id} alert_id={alert.id}: {e}")
            sent_flag = False
            send_reason = "email_failed"

    # =====================================================================
    # SECTION: RECORD RUN, UPDATE ALERT STATE
    # =====================================================================
    # =======================================
    # SECTION: FINALISE ALERT RUN ROW
    # Updates the run row we created at the start of processing.
    # =======================================

    try:
        run_row.price_found = current_price
        run_row.sent = bool(sent_flag)
        run_row.reason = str(send_reason)
        db.add(run_row)
    except Exception:
        # fallback update if the ORM instance is unavailable for any reason
        db.query(AlertRun).filter(AlertRun.id == alert_run_id).update(
            {
                "price_found": current_price,
                "sent": bool(sent_flag),
                "reason": str(send_reason),
            }
        )

    # Track best price on the alert record
    try:
        if stored_best_price is None or current_price < int(stored_best_price):
            alert.last_price = current_price
    except Exception:
        alert.last_price = current_price

    if sent_flag:
        alert.times_sent = (getattr(alert, "times_sent", None) or 0) + 1
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

        # -------------------------------------------------------
        # Expire alerts: if today is after the last departure date
        # Rule: expire when today > (departure_end or departure_start)
        # -------------------------------------------------------
        today = datetime.utcnow().date()
        now = datetime.utcnow()

        expiring = (
            db.query(Alert)
            .filter(Alert.is_active == True)  # noqa: E712
            .filter(func.coalesce(Alert.departure_end, Alert.departure_start).isnot(None))
            .filter(func.coalesce(Alert.departure_end, Alert.departure_start) < today)
            .all()
        )

        if expiring:
            for a in expiring:
                a.is_active = False
                a.updated_at = now
                # If you later add expired_at, this will auto-use it without breaking now
                if hasattr(a, "expired_at"):
                    try:
                        setattr(a, "expired_at", now)
                    except Exception:
                        pass
            db.commit()
            print(f"[alerts] Expired {len(expiring)} alerts (today={today})")

        alerts = db.query(Alert).filter(Alert.is_active == True).all()  # noqa: E712
        print(f"[alerts] Found {len(alerts)} active alerts before cadence filtering")

        eligible_alerts = []
        skipped = 0

        for a in alerts:
            try:
                user = _get_user_for_alert(db, a)

                # If user is missing, fail open and attempt the alert,
                # process_alert will record "no_user_for_alert"
                if not user:
                    eligible_alerts.append(a)
                    continue

                plan_tier = (getattr(user, "plan_tier", None) or "").lower()

                # Admin alerts always run
                if plan_tier == "admin":
                    eligible_alerts.append(a)
                    continue

                checks_per_day = int(getattr(user, "plan_checks_per_day", 1) or 1)

                last_run_at = getattr(a, "last_run_at", None)
                if last_run_at:
                    min_interval = _min_interval_seconds_for_checks_per_day(checks_per_day)
                    elapsed = (datetime.utcnow() - last_run_at).total_seconds()
                    if elapsed < min_interval:
                        skipped += 1
                        continue

                eligible_alerts.append(a)

            except Exception:
                # If anything is weird, fail open to avoid dropping alerts silently
                eligible_alerts.append(a)

        print(
            f"[alerts] Running alerts cycle for {len(eligible_alerts)} eligible alerts "
            f"(skipped {skipped} due to plan cadence)"
        )

        import traceback

        for alert in eligible_alerts:
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
    
@app.get("/debug-duffel-get")
def debug_duffel_get():
    # simple lightweight Duffel call that should always respond fast
    res = duffel_get("/air/airlines", params={"limit": 1})
    return {"ok": True, "type": str(type(res)), "sample": res}

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

# ---- Guardrails (in-memory, single-process) ----
# NOTE: If you ever run multiple workers/processes, these must move to Redis/queue.
import os
import time
import threading
from contextlib import contextmanager

MAX_CONCURRENT_SEARCHES = get_config_int("MAX_CONCURRENT_SEARCHES", 2)
SEARCH_HARD_CAP_SECONDS = get_config_int("SEARCH_HARD_CAP_SECONDS", 70)

_GLOBAL_SEARCH_SEM = threading.Semaphore(MAX_CONCURRENT_SEARCHES)
_USER_GUARD_LOCK = threading.Lock()
_USER_INFLIGHT = {}  # user_key -> {"job_id": str|None, "started_at": float}

def _user_key_from_params(params: SearchParams) -> str | None:
    """
    IMPORTANT: Per-user single-flight only works if this is truly stable.
    Do NOT fallback to origin/destination for async scans, it will fail to block overlaps reliably.
    """
    for k in ("user_external_id", "userExternalId", "user_email", "userEmail"):
        v = getattr(params, k, None)
        if v:
            return str(v).strip().lower()
    return None

# Hard cap flags, job_id -> cap hit
_HARD_CAP_HIT = set()  # set[str]

@contextmanager
def _hard_runtime_cap(seconds: int, job_id: str | None = None):
    """
    Hard cap: if work hangs, do NOT kill the whole web process.
    Instead, mark the job as capped and let the job loop cancel safely.
    """
    if not seconds or seconds <= 0:
        yield
        return

    def _flag():
        if job_id:
            try:
                _HARD_CAP_HIT.add(job_id)
            except Exception:
                pass
        print(f"[guardrail] HARD CAP HIT after {seconds}s job_id={job_id}")

    t = threading.Timer(seconds, _flag)
    t.daemon = True
    t.start()
    try:
        yield
    finally:
        t.cancel()

def _begin_user_inflight(user_key: str, job_id: str | None) -> bool:
    now = time.monotonic()
    with _USER_GUARD_LOCK:
        # TTL cleanup: if a lock is older than hard cap + buffer, drop it
        rec = _USER_INFLIGHT.get(user_key)
        if rec and (now - rec.get("started_at", now)) > (SEARCH_HARD_CAP_SECONDS + 30):
            print(f"[guardrail] stale_inflight_cleared user_key={user_key}")
            _USER_INFLIGHT.pop(user_key, None)

        if user_key in _USER_INFLIGHT:
            return False

        _USER_INFLIGHT[user_key] = {"job_id": job_id, "started_at": now}
        return True

def _end_user_inflight(user_key: str):
    with _USER_GUARD_LOCK:
        _USER_INFLIGHT.pop(user_key, None)

def _peek_user_inflight(user_key: str):
    """Return (job_id, age_seconds) for current inflight record, or (None, None)."""
    now = time.monotonic()
    with _USER_GUARD_LOCK:
        rec = _USER_INFLIGHT.get(user_key) or {}
        job_id = rec.get("job_id")
        started_at = rec.get("started_at")
    age = (now - started_at) if started_at else None
    return job_id, age

def _run_search_job_guarded(job_id: str, user_key: str):
    try:
        with _hard_runtime_cap(SEARCH_HARD_CAP_SECONDS, job_id=job_id):
            # IMPORTANT: call the actual job runner function that exists in this file
            _run_search_job(job_id)
    except Exception as e:
        # Ensure UI does not poll forever
        try:
            job = JOBS.get(job_id)
            if job:
                job.status = JobStatus.FAILED
                job.error = f"job_crash: {type(e).__name__}: {e}"
        except Exception as _e:
            print(f"[JOB {job_id}] FAILED status update failed: {_e}")
        raise
    finally:
        try:
            job = JOBS.get(job_id)
            st = getattr(job, "status", None)
            print(f"[JOB {job_id}] FINALLY reached status={st} user_key={user_key}")
        except Exception as _e:
            print(f"[JOB {job_id}] FINALLY reached log_failed: {_e}")

        _end_user_inflight(user_key)
        _GLOBAL_SEARCH_SEM.release()

@app.post("/ttn/book")
def ttn_book(payload: TTNBookRequest):
    """
    TTN booking probe.
    Phase 1 goal: discover the correct TTN booking/payment endpoint shape
    using session_id + recommendation_id.
    """
    api_key = _get_ttn_api_key()
    if not api_key:
        raise HTTPException(status_code=500, detail="TTN_API_KEY is not configured")

    print(f"[ttn] book.probe START session_id={payload.session_id} rec_id={payload.recommendation_id}")

    def _req(method: str, path: str, json_payload: Optional[dict] = None) -> dict:
        url = f"{TTN_BASE_URL}{path}"
        params = {"key": api_key}

        if method.upper() == "GET":
            params.update(
                {
                    "session_id": payload.session_id,
                    "recommendation_id": payload.recommendation_id,
                }
            )

        try:
            r = requests.request(
                method=method.upper(),
                url=url,
                params=params,
                json=json_payload if method.upper() != "GET" else None,
                headers=_ttn_headers(),
                timeout=30,
            )
        except Exception as e:
            return {
                "path": path,
                "method": method.upper(),
                "ok": False,
                "error": str(e),
            }

        body = (r.text or "")[:1200]
        out = {
            "path": path,
            "method": method.upper(),
            "status_code": r.status_code,
            "content_type": r.headers.get("content-type"),
            "body_preview": body,
        }

        if "application/json" in (out["content_type"] or ""):
            try:
                out["json"] = r.json()
            except Exception:
                pass

        return out

    candidates = [
        "/avia/book.json",
        "/avia/booking/create.json",
        "/avia/booking.json",
        "/avia/booking/start.json",
        "/avia/booking/confirm.json",
        "/avia/order/create.json",
        "/avia/order.json",
        "/avia/payment.json",
        "/avia/pay.json",
        "/avia/checkout.json",
        "/avia/ticketing.json",
        "/avia/issue.json",
    ]

    results = []
    for p in candidates:
        r_get = _req("GET", p)
        results.append(r_get)

        if r_get.get("status_code") and r_get["status_code"] != 404:
            print(f"[ttn] book.probe HIT method=GET path={p} status={r_get['status_code']}")
            return {"status": "probe", "hit": r_get, "all": results}

    # If we got here, everything was 404 (or no status_code)
    return {"status": "probe", "hit": None, "all": results}

@app.post("/ttn/checkout-link")
def ttn_checkout_link(payload: TTNCheckoutLinkRequest):
    api_key = _get_ttn_api_key()
    if not api_key:
        raise HTTPException(status_code=500, detail="TTN_API_KEY is not configured")

    auth_key = _get_ttn_auth_key()
    if not auth_key:
        raise HTTPException(status_code=500, detail="TTN_AUTH_KEY is not configured")

    if not payload.passengers:
        raise HTTPException(status_code=400, detail="passengers is required")

    # Build booking params using TTN documented query structure
    params = {
        "key": api_key,
        "session_id": payload.providerSessionId,
        "recommendation_id": payload.providerRecommendationId,
        "auth_key": auth_key,
        "real_email": payload.real_email,
        "real_phone": payload.real_phone,
    }

    for i, p in enumerate(payload.passengers):
        params[f"passengers[{i}][type]"] = p.type
        params[f"passengers[{i}][firstname]"] = p.firstname
        params[f"passengers[{i}][lastname]"] = p.lastname
        params[f"passengers[{i}][birthday]"] = p.birthday
        params[f"passengers[{i}][gender]"] = p.gender
        params[f"passengers[{i}][citizenship]"] = p.citizenship
        params[f"passengers[{i}][docnum]"] = p.docnum
        params[f"passengers[{i}][doc_expire_date]"] = p.doc_expire_date

    # 1) Create booking
    book_url = f"{TTN_BASE_URL}/avia/book.json"
    r = requests.get(book_url, params=params, headers=_ttn_headers(), timeout=30)

    ct = (r.headers.get("content-type") or "").lower()
    body_preview = (r.text or "")[:1200]

    if r.status_code != 200:
        raise HTTPException(
            status_code=502,
            detail={"stage": "book", "status_code": r.status_code, "content_type": ct, "body_preview": body_preview},
        )

    booking_id = None
    locator = None
    ttl = None

    book_json = None
    if "application/json" in ct:
        try:
            book_json = r.json()
        except Exception:
            book_json = None

    if isinstance(book_json, dict):
        resp = book_json.get("response") or {}
        booking = resp.get("booking") or {}
        locator = booking.get("locator")
        booking_id = booking.get("booking-id") or booking.get("booking_id")
        ttl = booking.get("ticketing-time-limit") or booking.get("ticketing_time_limit")

    # Fallback for XML responses
    if (not locator) and ("xml" in ct or (r.text or "").lstrip().startswith("<")):
        try:
            import xml.etree.ElementTree as ET
            root = ET.fromstring(r.text)
            b = root.find(".//booking")
            if b is not None:
                locator = b.get("locator") or locator
                booking_id = b.get("booking-id") or booking_id
                ttl = b.get("ticketing-time-limit") or ttl
        except Exception:
            pass

    if not locator:
        raise HTTPException(
            status_code=502,
            detail={"stage": "book_parse", "message": "Could not extract locator from booking response", "body_preview": body_preview},
        )

    # 2) Generate FAT redirect
    # TTN Postman calls this "Request for Redirect payment". Endpoint name varies, so try a small safe list.
    redirect_candidates = [
        "/payment/redirect.json",
        "/payment/redirect.xml",
        "/payment/redirect",
        "/avia/payment/redirect.json",
        "/avia/payment/redirect.xml",
    ]

    redirect_result = None
    for path in redirect_candidates:
        url = f"{TTN_BASE_URL}{path}"
        rp = {
            "key": api_key,
            "auth_key": auth_key,
            "service": "avia",
            "order_id": locator,
        }
        rr = requests.get(url, params=rp, headers=_ttn_headers(), timeout=30)
        rct = (rr.headers.get("content-type") or "").lower()
        rbody = (rr.text or "")[:1200]

        if rr.status_code == 404:
            continue

        # Try JSON
        if "application/json" in rct:
            try:
                j = rr.json()
            except Exception:
                j = None
            if isinstance(j, dict):
                # common patterns: response.link.url, response.url, link.url
                resp = j.get("response") or {}
                link = resp.get("link") or j.get("link") or {}
                checkout_url = link.get("url") or resp.get("url") or j.get("url")
                if checkout_url:
                    redirect_result = {"checkout_url": checkout_url, "raw": j, "path": path}
                    break

        # Try XML
        if redirect_result is None and ("xml" in rct or (rr.text or "").lstrip().startswith("<")):
            try:
                import xml.etree.ElementTree as ET
                root = ET.fromstring(rr.text)
                link = root.find(".//link")
                if link is not None and link.get("url"):
                    redirect_result = {"checkout_url": link.get("url"), "raw": rbody, "path": path}
                    break
            except Exception:
                pass

        # If this endpoint returns something non-404 but no url, keep a preview for debugging
        redirect_result = {"error": "no_checkout_url_in_response", "path": path, "content_type": rct, "body_preview": rbody}

    if not redirect_result or not redirect_result.get("checkout_url"):
        raise HTTPException(
            status_code=502,
            detail={"stage": "redirect", "message": "Could not generate FAT redirect URL", "locator": locator, "probe": redirect_result},
        )

    return {
        "status": "ok",
        "locator": locator,
        "booking_id": booking_id,
        "ticketing_time_limit": ttl,
        "checkout_url": redirect_result["checkout_url"],
        "redirect_path": redirect_result.get("path"),
    }

@app.post("/search-business")
def search_business(params: SearchParams, background_tasks: BackgroundTasks):
    request_id = str(uuid4())

    if not _get_ttn_api_key():
        return {"status": "error", "source": "ttn_not_configured", "options": []}

    try:
        print(f"[guardrail] request_keys={sorted(list(params.model_dump().keys()))}")
    except Exception:
        pass

    print(
        f"[search_business] search_mode={getattr(params,'search_mode',None)} "
        f"earliestDeparture={getattr(params,'earliestDeparture',None)} "
        f"latestDeparture={getattr(params,'latestDeparture',None)} "
        f"nights={getattr(params,'nights',None)} "
        f"minStayDays={getattr(params,'minStayDays',None)} "
        f"maxStayDays={getattr(params,'maxStayDays',None)}"
    )

    max_passengers = get_config_int("MAX_PASSENGERS", 4)
    if params.passengers > max_passengers:
        params.passengers = max_passengers

    default_cabin = get_config_str("DEFAULT_CABIN", "BUSINESS") or "BUSINESS"
    if not params.cabin:
        params.cabin = default_cabin

    user_key = _user_key_from_params(params)
    estimated_pairs = estimate_date_pairs(params)

    inflight_job_id = None
    inflight_age = None
    if user_key:
        inflight_job_id, inflight_age = _peek_user_inflight(user_key)

        print(
            f"[trace] request_id={request_id} user_key={repr(user_key)} "
            f"estimated_pairs={estimated_pairs} inflight_job_id={inflight_job_id} "
            f"inflight_age_s={None if inflight_age is None else int(inflight_age)} "
            f"origin={getattr(params,'origin',None)} dest={getattr(params,'destination',None)} "
            f"search_mode={getattr(params,'search_mode',None)} "
            f"earliestDeparture={getattr(params,'earliestDeparture',None)} "
            f"latestDeparture={getattr(params,'latestDeparture',None)} "
            f"nights={getattr(params,'nights',None)}"
        )

        print(
            f"[guardrail] user_external_id_raw={repr(getattr(params, 'user_external_id', None))} user_key={repr(user_key)}"
        )

    # If we cannot identify the user, allow only sync (single-pair) searches.
    # This prevents anonymous refresh spam from starting multiple async jobs.
    if not user_key and estimated_pairs > 1:
        print(
            f"[guardrail] missing_user_id "
            f"user_external_id_raw={repr(getattr(params, 'user_external_id', None))} "
            f"userExternalId_raw={repr(getattr(params, 'userExternalId', None))} "
            f"user_email_raw={repr(getattr(params, 'user_email', None))} "
            f"userEmail_raw={repr(getattr(params, 'userEmail', None))} "
            f"origin={params.origin} dest={params.destination} "
            f"earliest={params.earliestDeparture} latest={params.latestDeparture} "
            f"minStay={params.minStayDays} maxStay={params.maxStayDays}"
        )

        return {
            "status": "error",
            "source": "missing_user_id",
            "message": "Missing user identity for multi-date searches. Please sign in and retry.",
        }

    # ---- Per-user single-flight guard (REUSE mode) ----
    # If the user already has an inflight async job, return its jobId instead of starting a new scan.
    new_job_id = None

    if user_key:
        inflight_job_id, inflight_age = _peek_user_inflight(user_key)

        if inflight_job_id:
            j = JOBS.get(inflight_job_id)
            if j and j.status in (JobStatus.PENDING, JobStatus.RUNNING):
                print(
                    f"[guardrail] reuse_inflight user_key={user_key} job_id={inflight_job_id} "
                    f"age_s={None if inflight_age is None else int(inflight_age)}"
                )
                return {
                    "status": "ok",
                    "mode": "async",
                    "jobId": inflight_job_id,
                    "message": "Search already running, reusing existing job",
                }

        # Only lock inflight for multi-pair (async) searches.
        if estimated_pairs > 1:
            new_job_id = str(uuid4())

            begin_ok = _begin_user_inflight(user_key, job_id=new_job_id)
            if not begin_ok:
                blocker_job_id, blocker_age = _peek_user_inflight(user_key)

                print(
                    f"[trace] request_id={request_id} begin_ok=False user_key={repr(user_key)} "
                    f"blocker_job_id={blocker_job_id} "
                    f"blocker_age_s={None if blocker_age is None else int(blocker_age)}"
                )

                if blocker_job_id:
                    print(f"[guardrail] returning_existing_job user_key={user_key} job_id={blocker_job_id}")
                    return {
                        "status": "ok",
                        "mode": "async",
                        "jobId": blocker_job_id,
                        "message": "Search already running, returning existing job.",
                    }

                print(f"[guardrail] search_in_progress user_key={user_key}")
                return {
                    "status": "error",
                    "source": "search_in_progress",
                    "message": "A search is already running for this user, please wait for it to finish.",
                }

            print(f"[trace] request_id={request_id} begin_ok=True user_key={repr(user_key)} job_id={new_job_id}")

    # ---- Global concurrency guard ----
    acquired = _GLOBAL_SEARCH_SEM.acquire(blocking=False)

    if not acquired:
        if user_key:
            _end_user_inflight(user_key)
        print(f"[guardrail] server_busy user_key={user_key}")
        return {
            "status": "error",
            "source": "server_busy",
            "message": "Server is busy running other searches, please retry in a moment.",
        }

    try:
        # ---- Sync path (single pair) ----
        if estimated_pairs <= 1:
            with _hard_runtime_cap(SEARCH_HARD_CAP_SECONDS, job_id=None):
                options = run_ttn_scan(params)
                options = apply_global_airline_cap(options, max_share=0.5)
                return {
                    "status": "ok",
                    "mode": "sync",
                    "source": "ttn",
                    "options": [o.dict() for o in options],
                }

        # ---- Async path (multi pair) ----
        job_id = new_job_id or str(uuid4())
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

        if user_key:
            with _USER_GUARD_LOCK:
                if user_key in _USER_INFLIGHT:
                    _USER_INFLIGHT[user_key]["job_id"] = job_id

                t = threading.Thread(
                    target=_run_search_job_guarded,
                    args=(job_id, user_key),
                    daemon=True,
                    name=f"search-job-{job_id[:8]}",
                )
                t.start()
        else:
            # This should not happen due to the guard above, but keep it safe.
            _GLOBAL_SEARCH_SEM.release()
            return {
                "status": "error",
                "source": "missing_user_id",
                "message": "Missing user identity for multi-date searches. Please sign in and retry.",
            }

        return {"status": "ok", "mode": "async", "jobId": job_id, "message": "Search started"}

    except Exception as e:
        if user_key:
            _end_user_inflight(user_key)
        _GLOBAL_SEARCH_SEM.release()
        raise e

    finally:
        # For sync we release here.
        # For async the guarded background task releases.
        if estimated_pairs <= 1:
            if user_key:
                _end_user_inflight(user_key)
            _GLOBAL_SEARCH_SEM.release()

# =====================================================================
# SECTION END: MAIN SEARCH ROUTES
# =====================================================================

# =====================================================================
# SECTION START: SEARCH STATUS AND RESULTS ROUTES
# =====================================================================

# Average seconds per date pair (based on Duffel API response times)
ESTIMATED_SECONDS_PER_PAIR = 6.0

@app.get("/search-status/{job_id}", response_model=SearchStatusResponse)
def get_search_status(job_id: str, preview_limit: int = 20):
    job = JOBS.get(job_id)
    
    if not job:
        return SearchStatusResponse(
            jobId=job_id,
            status=JobStatus.CANCELLED,
            processedPairs=0,
            totalPairs=0,
            progress=0.0,
            error="Job not found in memory, the server likely restarted. Please start a new search.",
            previewCount=0,
            previewOptions=[],
            elapsedSeconds=None,
            estimatedTotalSeconds=None,
            estimatedProgressPct=None,
        )
    
    options = JOB_RESULTS.get(job_id, [])
    preview = options[:preview_limit] if preview_limit > 0 else []
    
    total_pairs = job.total_pairs or 0
    processed_pairs = job.processed_pairs or 0
    print(f"[status] job_id={job_id[:8]} status={job.status} processed={processed_pairs}/{total_pairs}")
    progress = float(processed_pairs) / float(total_pairs) if total_pairs > 0 else 0.0
    
    # =====================================================================
    # ESTIMATED PROGRESS CALCULATION
    # =====================================================================
    elapsed_seconds = None
    estimated_total_seconds = None
    estimated_progress_pct = None
    
    if job.status == JobStatus.RUNNING and total_pairs > 0:
        now = datetime.utcnow()
        elapsed_seconds = (now - job.created_at).total_seconds()
        estimated_total_seconds = total_pairs * ESTIMATED_SECONDS_PER_PAIR
        
        # Time-based progress (capped at 95% to avoid showing 100% before done)
        time_based_pct = min(95.0, (elapsed_seconds / estimated_total_seconds) * 100) if estimated_total_seconds > 0 else 0.0
        
        # Use the HIGHER of time-based or actual progress
        actual_pct = progress * 100
        estimated_progress_pct = max(time_based_pct, actual_pct)
        
        # Recalculate estimate based on actual performance
        if processed_pairs > 0:
            actual_seconds_per_pair = elapsed_seconds / processed_pairs
            estimated_total_seconds = total_pairs * actual_seconds_per_pair
    
    elif job.status == JobStatus.COMPLETED:
        elapsed_seconds = (job.updated_at - job.created_at).total_seconds() if job.updated_at else None
        estimated_total_seconds = elapsed_seconds
        estimated_progress_pct = 100.0
    
    return SearchStatusResponse(
        jobId=job.id,
        status=job.status,
        processedPairs=processed_pairs,
        totalPairs=total_pairs,
        progress=progress,
        error=job.error,
        previewCount=len(preview),
        previewOptions=preview,
        elapsedSeconds=round(elapsed_seconds, 1) if elapsed_seconds is not None else None,
        estimatedTotalSeconds=round(estimated_total_seconds, 1) if estimated_total_seconds is not None else None,
        estimatedProgressPct=round(estimated_progress_pct, 1) if estimated_progress_pct is not None else None,
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
# SECTION START: ALERT RUN SNAPSHOT (READ ONLY)
# =====================================================================

@app.get("/alert-run-snapshot/{alert_run_id}")
def get_alert_run_snapshot(
    alert_run_id: str,
    x_user_id: str = Header(None, alias="X-User-Id"),
    x_admin_token: str = Header(None, alias="X-Admin-Token"),
):
    # Validate UUID early
    try:
        run_uuid = str(UUID(alert_run_id))
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid alert_run_id, must be a UUID")

    # Allow admin override, otherwise require user id and enforce ownership
    received = (x_admin_token or "").strip()
    expected = (ADMIN_API_TOKEN or "").strip()
    if received.lower().startswith("bearer "):
        received = received[7:].strip()
    is_admin = (expected != "") and (received == expected)

    if not is_admin and not x_user_id:
        raise HTTPException(status_code=401, detail="Missing X-User-Id")

    db = SessionLocal()
    try:
        sql = """
            SELECT
                ar.id                   AS alert_run_id,
                ar.alert_id             AS alert_id,
                ar.run_at               AS created_at,
                ars.best_price_per_pax  AS best_price_per_pax,
                ars.currency            AS currency,
                ars.params              AS params,
                ars.top_results         AS top_results,
                ars.meta                AS meta
            FROM alert_runs ar
            JOIN alerts a
              ON a.id = ar.alert_id
            JOIN alert_run_snapshots ars
              ON ars.alert_run_id = ar.id
            WHERE ar.id = :rid
        """

        bind = {"rid": run_uuid}

        if not is_admin:
            sql += " AND a.user_external_id = :uid"
            bind["uid"] = x_user_id

        sql += " ORDER BY ars.created_at DESC LIMIT 1"

        row = (
            db.execute(text(sql), bind)
            .mappings()
            .first()
        )

        if not row:
            raise HTTPException(status_code=404, detail="Snapshot not found for this alertRunId")

        params = row["params"] or {}
        meta = row["meta"] or {}
        top_results = row["top_results"] or []

        # Passengers for per-pax normalization (defensive for older snapshots)
        passengers = 1
        try:
            passengers = int(params.get("passengers") or 1)
        except Exception:
            passengers = 1
        if passengers <= 0:
            passengers = 1

        def _norm_item_price(item: dict) -> dict:
            if not isinstance(item, dict):
                return item

            # If already per-pax, do nothing
            if "price_per_pax" in item and item["price_per_pax"] is not None:
                return item

            # Prefer explicit total keys
            for total_key in ("total_price", "totalPrice", "price_total", "priceTotal"):
                if total_key in item and item[total_key] is not None:
                    try:
                        total = float(item[total_key])
                        item["price_per_pax"] = round(total / passengers, 2)
                        return item
                    except Exception:
                        return item

            return item

        if isinstance(top_results, list):
            top_results = [_norm_item_price(r) for r in top_results]

        meta = dict(meta)
        meta["snapshot_mode"] = True
        meta["passengers"] = passengers

        return {
            "alert_run_id": str(row["alert_run_id"]),
            "alert_id": row["alert_id"],
            "created_at": row["created_at"],
            "best_price_per_pax": row["best_price_per_pax"],
            "currency": row["currency"],
            "params": params,
            "top_results": top_results,
            "meta": meta,
        }
    finally:
        db.close()

# =====================================================================
# SECTION END: ALERT RUN SNAPSHOT (READ ONLY)
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
# SECTION START: ADMIN FORCE SYNC ENDPOINT
# =====================================================================

class AdminSyncRequest(BaseModel):
    email: Optional[str] = None
    external_id: Optional[str] = None
    plan_tier_code: str  # required: free, gold, platinum, tester, admin


@app.post("/admin/sync-user-tier")
def admin_sync_user_tier(
    payload: AdminSyncRequest,
    x_user_id: Optional[str] = Header(None, alias="X-User-Id"),
):
    """
    Admin-only endpoint to force-sync a user's plan tier.
    Call this after changing a user's tier in Base44 admin panel.
    """
    db = SessionLocal()
    try:
        # 1) Auth: require admin
        if not x_user_id:
            raise HTTPException(status_code=401, detail="X-User-Id header required")
        
        admin_user = db.query(AppUser).filter(AppUser.external_id == x_user_id).first()
        if not admin_user or admin_user.plan_tier != "admin":
            raise HTTPException(status_code=403, detail="Admin access required")

        # 2) Find target user
        if not payload.email and not payload.external_id:
            raise HTTPException(status_code=400, detail="Must provide email or external_id")

        if payload.external_id:
            target_user = db.query(AppUser).filter(AppUser.external_id == payload.external_id).first()
        else:
            target_user = db.query(AppUser).filter(func.lower(AppUser.email) == payload.email.lower()).first()

        if not target_user:
            raise HTTPException(status_code=404, detail="User not found")

        # 3) Plan defaults
        PLAN_DEFAULTS = {
            "free": {"plan_tier": "free", "plan_active_alert_limit": 1, "plan_max_departure_window_days": 7, "plan_checks_per_day": 3},
            "gold": {"plan_tier": "gold", "plan_active_alert_limit": 3, "plan_max_departure_window_days": 14, "plan_checks_per_day": 6},
            "platinum": {"plan_tier": "platinum", "plan_active_alert_limit": 10, "plan_max_departure_window_days": 30, "plan_checks_per_day": 12},
            "tester": {"plan_tier": "tester", "plan_active_alert_limit": 10000, "plan_max_departure_window_days": 365, "plan_checks_per_day": 10000},
            "admin": {"plan_tier": "admin", "plan_active_alert_limit": 10000, "plan_max_departure_window_days": 365, "plan_checks_per_day": 10000},
        }

        tier_norm = payload.plan_tier_code.strip().lower()
        if tier_norm not in PLAN_DEFAULTS:
            raise HTTPException(status_code=400, detail=f"Invalid tier: {tier_norm}. Must be one of: {list(PLAN_DEFAULTS.keys())}")

        # 4) Apply update
        defaults = PLAN_DEFAULTS[tier_norm]
        target_user.plan_tier = defaults["plan_tier"]
        target_user.plan_active_alert_limit = defaults["plan_active_alert_limit"]
        target_user.plan_max_departure_window_days = defaults["plan_max_departure_window_days"]
        target_user.plan_checks_per_day = defaults["plan_checks_per_day"]

        db.commit()
        db.refresh(target_user)

        return {
            "status": "ok",
            "user": {
                "email": target_user.email,
                "external_id": target_user.external_id,
                "plan_tier": target_user.plan_tier,
                "plan_active_alert_limit": target_user.plan_active_alert_limit,
                "plan_max_departure_window_days": target_user.plan_max_departure_window_days,
                "plan_checks_per_day": target_user.plan_checks_per_day,
            }
        }

    finally:
        db.close()

# =====================================================================
# SECTION END: ADMIN FORCE SYNC ENDPOINT
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
    print(f"[user-sync] payload: email={payload.email}, plan_tier_code={payload.plan_tier_code}, external_id={getattr(payload, 'external_id', None) or getattr(payload, 'id', None)}")
    db = SessionLocal()
    canonical_external_id = None

    try:
        # ------------------------------
        # Canonical identity
        # ------------------------------
        canonical_external_id = (
            (getattr(payload, "external_id", None) or "").strip()
            or (getattr(payload, "id", None) or "").strip()
            or (getattr(payload, "user_id", None) or "").strip()
        )
        if not canonical_external_id:
            raise HTTPException(status_code=400, detail="Missing external_id")

        # ------------------------------
        # Plan defaults: single source of truth
        # ------------------------------
        FREE_DEFAULTS = {
            "plan_tier": "free",
            "plan_active_alert_limit": 1,
            "plan_max_departure_window_days": 7,
            "plan_checks_per_day": 3,
        }

        TESTER_DEFAULTS = {
            "plan_tier": "tester",
            "plan_active_alert_limit": 10_000,        # effectively unlimited
            "plan_max_departure_window_days": 365,    # effectively unlimited
            "plan_checks_per_day": 10_000,            # effectively unlimited
        }

        GOLD_DEFAULTS = {
            "plan_tier": "gold",
            "plan_active_alert_limit": 3,
            "plan_max_departure_window_days": 14,
            "plan_checks_per_day": 6,
        }

        PLATINUM_DEFAULTS = {
            "plan_tier": "platinum",
            "plan_active_alert_limit": 10,
            "plan_max_departure_window_days": 30,
            "plan_checks_per_day": 12,
        }

        ALLOWED_TIERS = {"free", "gold", "platinum", "tester", "admin"}

        # 1) Primary lookup: external id
        user = (
            db.query(AppUser)
            .filter(AppUser.external_id == canonical_external_id)
            .first()
        )

        # 2) Secondary lookup: same email, new external_id
        if user is None and payload.email:
            email_norm = payload.email.strip().lower()
            user = (
                db.query(AppUser)
                .filter(func.lower(AppUser.email) == email_norm)
                .first()
            )
            if user is not None:
                user.external_id = canonical_external_id

        if user is None:
            # 3) Create only if neither external_id nor email exists
            user = AppUser(
                external_id=canonical_external_id,
                email=(payload.email.strip().lower() if payload.email else None),
                first_name=payload.first_name,
                last_name=payload.last_name,
                country=payload.country,
                marketing_consent=payload.marketing_consent,
                source=payload.source or "base44",
            )
            db.add(user)
        else:
            # 4) Update profile fields (do not downgrade plan entitlements here)
            if payload.email:
                user.email = payload.email.strip().lower()
            user.first_name = payload.first_name
            user.last_name = payload.last_name
            user.country = payload.country
            user.marketing_consent = payload.marketing_consent
            user.source = payload.source or user.source

        # Adopt tier if provided, but never overwrite admin
        if payload.plan_tier_code:
            tier_norm = payload.plan_tier_code.strip().lower()
            print(f"[user-sync] tier_norm={tier_norm}, current_tier={user.plan_tier}, in_allowed={tier_norm in ALLOWED_TIERS}")
            if tier_norm in ALLOWED_TIERS:
                if user.plan_tier != "admin":
                    old_tier = user.plan_tier
                    user.plan_tier = tier_norm
                    print(f"[user-sync] UPDATED tier: {old_tier} -> {user.plan_tier}")

        # 5) Ensure entitlements exist, and lock plan values where required
        if getattr(user, "plan_tier", None) in (None, ""):
            user.plan_tier = FREE_DEFAULTS["plan_tier"]

        if getattr(user, "plan_active_alert_limit", None) is None:
            user.plan_active_alert_limit = FREE_DEFAULTS["plan_active_alert_limit"]

        if getattr(user, "plan_max_departure_window_days", None) is None:
            user.plan_max_departure_window_days = FREE_DEFAULTS["plan_max_departure_window_days"]

        if getattr(user, "plan_checks_per_day", None) is None:
            user.plan_checks_per_day = FREE_DEFAULTS["plan_checks_per_day"]

        # Lock plan values
        if user.plan_tier == "free":
            user.plan_active_alert_limit = FREE_DEFAULTS["plan_active_alert_limit"]
            user.plan_max_departure_window_days = FREE_DEFAULTS["plan_max_departure_window_days"]
            user.plan_checks_per_day = FREE_DEFAULTS["plan_checks_per_day"]

        elif user.plan_tier == "gold":
            user.plan_active_alert_limit = GOLD_DEFAULTS["plan_active_alert_limit"]
            user.plan_max_departure_window_days = GOLD_DEFAULTS["plan_max_departure_window_days"]
            user.plan_checks_per_day = GOLD_DEFAULTS["plan_checks_per_day"]

        elif user.plan_tier == "platinum":
            user.plan_active_alert_limit = PLATINUM_DEFAULTS["plan_active_alert_limit"]
            user.plan_max_departure_window_days = PLATINUM_DEFAULTS["plan_max_departure_window_days"]
            user.plan_checks_per_day = PLATINUM_DEFAULTS["plan_checks_per_day"]

        elif user.plan_tier == "tester":
            user.plan_active_alert_limit = TESTER_DEFAULTS["plan_active_alert_limit"]
            user.plan_max_departure_window_days = TESTER_DEFAULTS["plan_max_departure_window_days"]
            user.plan_checks_per_day = TESTER_DEFAULTS["plan_checks_per_day"]

        print(f"[user-sync] ABOUT TO COMMIT - tier={user.plan_tier}, limit={user.plan_active_alert_limit}")
        try:
            db.commit()
            print(f"[user-sync] COMMITTED - tier is now {user.plan_tier}")
        except Exception as commit_error:
            print(f"[user-sync] COMMIT FAILED: {commit_error}")
            import traceback
            print(f"[user-sync] COMMIT TRACEBACK: {traceback.format_exc()}")
            raise
        db.refresh(user)
        print(f"[user-sync] AFTER REFRESH - tier is {user.plan_tier}")

        db.commit()
        print(f"[user-sync] COMMITTED - tier is now {user.plan_tier}")
        db.refresh(user)
        print(f"[user-sync] AFTER REFRESH - tier is {user.plan_tier}")
        return {"status": "ok", "id": user.id}

    except Exception:
        db.rollback()

        # If a race occurs, return the existing user if possible
        if canonical_external_id:
            existing = (
                db.query(AppUser)
                .filter(AppUser.external_id == canonical_external_id)
                .first()
            )
            if existing is not None:
                return {"status": "ok", "id": existing.id}

        if payload.email:
            email_norm = payload.email.strip().lower()
            existing = (
                db.query(AppUser)
                .filter(func.lower(AppUser.email) == email_norm)
                .first()
            )
            if existing is not None:
                return {"status": "ok", "id": existing.id}

        raise

    finally:
        db.close()
from fastapi import Request

@app.post("/base44/user-webhook")
async def base44_user_webhook(request: Request, x_webhook_secret: str = Header(None, alias="X-Webhook-Secret")):
    # Basic shared-secret auth
    expected = os.getenv("BASE44_WEBHOOK_SECRET")
    if expected and x_webhook_secret != expected:
        raise HTTPException(status_code=401, detail="Invalid webhook secret")

    payload = await request.json()

    # Base44 payload shapes can vary, support a few common ones
    user_obj = None
    if isinstance(payload, dict):
        # Possible shapes:
        # 1) { "event": "...", "data": { "user": {...} } }
        # 2) { "type": "...", "user": {...} }
        # 3) { "user": {...} }
        # 4) { ...user fields directly... }
        if isinstance(payload.get("data"), dict) and isinstance(payload["data"].get("user"), dict):
            user_obj = payload["data"]["user"]
        elif isinstance(payload.get("user"), dict):
            user_obj = payload["user"]
        else:
            user_obj = payload

    if not isinstance(user_obj, dict):
        raise HTTPException(status_code=400, detail="Invalid payload")

    external_id = user_obj.get("id") or user_obj.get("external_id") or user_obj.get("user_id")
    email = (user_obj.get("email") or "").strip().lower()

    if not external_id or not email:
        raise HTTPException(status_code=400, detail="Missing id or email")

    first_name = user_obj.get("first_name")
    last_name = user_obj.get("last_name")
    full_name = user_obj.get("full_name") or user_obj.get("name")
    if full_name and (not first_name and not last_name):
        parts = str(full_name).strip().split()
        if parts:
            first_name = parts[0]
            last_name = " ".join(parts[1:]) if len(parts) > 1 else None

    plan_tier_code = (user_obj.get("plan_tier_code") or "free").strip().lower()

    # Plan defaults: single source of truth
    PLAN_DEFAULTS = {
        "free": {"plan_tier": "free", "plan_active_alert_limit": 1, "plan_max_departure_window_days": 7, "plan_checks_per_day": 3},
        "gold": {"plan_tier": "gold", "plan_active_alert_limit": 3, "plan_max_departure_window_days": 14, "plan_checks_per_day": 6},
        "platinum": {"plan_tier": "platinum", "plan_active_alert_limit": 10, "plan_max_departure_window_days": 30, "plan_checks_per_day": 12},
        "tester": {"plan_tier": "tester", "plan_active_alert_limit": 10_000, "plan_max_departure_window_days": 365, "plan_checks_per_day": 10_000},
        "admin": {"plan_tier": "admin", "plan_active_alert_limit": 10_000, "plan_max_departure_window_days": 365, "plan_checks_per_day": 10_000},
    }

    defaults = PLAN_DEFAULTS.get(plan_tier_code, PLAN_DEFAULTS["free"])

    db = SessionLocal()
    try:
        user = db.query(AppUser).filter(AppUser.external_id == external_id).first()

        if user is None:
            user = db.query(AppUser).filter(func.lower(AppUser.email) == email).first()
            if user is not None:
                user.external_id = external_id

        if user is None:
            user = AppUser(
                external_id=external_id,
                email=email,
                first_name=first_name,
                last_name=last_name,
                source="base44-webhook",
                marketing_consent=user_obj.get("marketing_consent"),
                country=user_obj.get("country"),
            )
            db.add(user)
        else:
            user.email = email
            user.first_name = first_name
            user.last_name = last_name
            user.country = user_obj.get("country") or user.country
            if "marketing_consent" in user_obj:
                user.marketing_consent = user_obj.get("marketing_consent")
            user.source = "base44-webhook"

        # Lock plan values from Base44
        user.plan_tier = defaults["plan_tier"]
        user.plan_active_alert_limit = defaults["plan_active_alert_limit"]
        user.plan_max_departure_window_days = defaults["plan_max_departure_window_days"]
        user.plan_checks_per_day = defaults["plan_checks_per_day"]

        db.commit()
        return {"status": "ok", "external_id": external_id, "email": email, "plan_tier": user.plan_tier}

    except Exception:
        db.rollback()
        raise
    finally:
        db.close()
        
@app.get("/profile", response_model=ProfileResponse)
def get_profile(x_user_id: str = Header(..., alias="X-User-Id")):
    wallet_balance = USER_WALLETS.get(x_user_id, 0)

    app_user = None
    active_alerts = 0

    # Identity fields the UI expects from /profile
    display_name = "Member"
    external_id = x_user_id
    joined_at = None

    # Keep all DB access inside the session lifecycle
    db = SessionLocal()
    try:
        app_user = db.query(AppUser).filter(AppUser.external_id == x_user_id).first()

        if app_user:
            external_id = app_user.external_id
            joined_at = app_user.created_at

            # Count alerts by current ownership (user_email)
            active_alerts = (
                db.query(Alert)
                .filter(
                    ((Alert.user_external_id == app_user.external_id) |
                     (Alert.user_email == app_user.email)),
                    Alert.is_active == True,  # noqa: E712
                )
                .count()
            )
    
    finally:
        db.close()

    if app_user:
        email = app_user.email or ""

        limit = int(app_user.plan_active_alert_limit or 1)
        remaining = max(0, limit - int(active_alerts))

        entitlements = ProfileEntitlements(
            plan_tier=app_user.plan_tier or "free",
            active_alert_limit=limit,
            max_departure_window_days=int(app_user.plan_max_departure_window_days or 15),
            checks_per_day=int(app_user.plan_checks_per_day or 1),
        )

        alert_usage = ProfileAlertUsage(
            active_alerts=int(active_alerts),
            remaining_slots=int(remaining),
        )
    else:
        email = ""
        entitlements = None
        alert_usage = None

    profile_user = ProfileUser(id=external_id, email=email, credits=int(wallet_balance))
    subscription = SubscriptionInfo(
        plan="Flyyv " + (entitlements.plan_tier.capitalize() if entitlements else "Free"),
        status="active",
        renews_on=None,
    )
    wallet = WalletInfo(balance=int(wallet_balance), currency="credits")

    return ProfileResponse(
        display_name=display_name,
        external_id=external_id,
        joined_at=joined_at,
        user=profile_user,
        subscription=subscription,
        wallet=wallet,
        entitlements=entitlements,
        alertUsage=alert_usage,
    )

# =====================================================================
# SECTION END: PUBLIC CONFIG, USER SYNC, PROFILE
# =====================================================================

# =====================================================================
# SECTION START: ALERT ROUTES
# =====================================================================

def _alert_state(a: Alert) -> str:
    # expired if today is after the last departure date
    today = datetime.utcnow().date()
    end = a.departure_end or a.departure_start
    if end and today > end:
        return "expired"
    return "active" if a.is_active else "paused"


@app.post("/alerts", response_model=AlertOut)
def create_alert(payload: AlertCreate, x_user_id: str = Header(..., alias="X-User-Id")):
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

        # Plan enforcement: active alert limit (create)
        app_user = db.query(AppUser).filter(AppUser.external_id == x_user_id).first()
        if not app_user:
            raise HTTPException(status_code=401, detail={"code": "UNAUTHORIZED"})

        active_alerts = (
            db.query(Alert)
            .filter(
                ((Alert.user_external_id == x_user_id) | (Alert.user_email == app_user.email)),
                Alert.is_active == True,  # noqa: E712
            )
            .count()
        )

        limit = int(getattr(app_user, "plan_active_alert_limit", 1) or 1)

        # Creating an alert always creates it active in v1
        if active_alerts >= limit:
            raise HTTPException(status_code=403, detail={"code": "ALERT_LIMIT_REACHED"})

        # Plan enforcement: departure window limit (create)
        window_limit = int(getattr(app_user, "plan_max_departure_window_days", 15) or 15)
        if payload.departure_start and payload.departure_end:
            window_days = (payload.departure_end - payload.departure_start).days + 1
            if window_days > window_limit:
                raise HTTPException(status_code=403, detail={"code": "WINDOW_LIMIT_EXCEEDED"})

        if pax > max_passengers:
            pax = max_passengers

        dep_start = payload.departure_start
        dep_end = payload.departure_end or dep_start

        # Ensure DB-safe default for alert_type, Postgres column is NOT NULL
        resolved_alert_type = (
            "under_price"
            if payload.max_price is not None
            else (payload.alert_type or "new_best")
        )

        alert = Alert(
            id=alert_id,
            user_email=app_user.email,
            user_external_id=x_user_id,
            origin=payload.origin,
            destination=payload.destination,
            cabin=payload.cabin,
            search_mode=search_mode_value,
            departure_start=dep_start,
            departure_end=dep_end,
            return_start=payload.return_start,
            return_end=payload.return_end,
            alert_type=resolved_alert_type,
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
            email=app_user.email,
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
            state=_alert_state(alert),
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


from fastapi import Request  # keep near other fastapi imports if not already present


@app.get("/alerts", response_model=List[AlertOut])
def get_alerts(
    request: Request,
    x_user_id: Optional[str] = Header(None, alias="X-User-Id"),
    include_inactive: bool = False,
):
    db = SessionLocal()
    try:
        # Hard reject legacy email-based access, header-only identity
        if "email" in request.query_params:
            raise HTTPException(
                status_code=400,
                detail="Email query param is not supported, use X-User-Id header",
            )

        if not x_user_id:
            raise HTTPException(status_code=401, detail="X-User-Id header required")

        app_user = db.query(AppUser).filter(AppUser.external_id == x_user_id).first()
        if not app_user:
            raise HTTPException(status_code=403, detail="Unknown user")

        query = db.query(Alert).filter(
            (Alert.user_external_id == x_user_id) | (Alert.user_email == app_user.email)
        )

        if not include_inactive:
            query = query.filter(Alert.is_active == True)  # noqa: E712

        alerts = query.order_by(Alert.created_at.desc()).all()
        result: List[AlertOut] = []

        alert_ids = [a.id for a in alerts]
        best_price_by_alert_id = {}

        if alert_ids:
            best_runs = (
                db.query(AlertRun.alert_id, func.min(AlertRun.price_found).label("best_price"))
                .filter(AlertRun.alert_id.in_(alert_ids))
                .filter(AlertRun.price_found.isnot(None))
                .group_by(AlertRun.alert_id)
                .all()
            )
            best_price_by_alert_id = {r.alert_id: r.best_price for r in best_runs}

        for a in alerts:
            result.append(
                AlertOut(
                    id=a.id,
                    email=getattr(app_user, "email", None),
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
                    state=_alert_state(a),
                    last_price=a.last_price,
                    best_price=best_price_by_alert_id.get(a.id),
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
        # Auth, resolve user, do not trust email param
        if not x_user_id:
            raise HTTPException(status_code=401, detail={"code": "UNAUTHORIZED"})

        app_user = db.query(AppUser).filter(AppUser.external_id == x_user_id).first()
        if not app_user:
            raise HTTPException(status_code=403, detail="Unknown user")

        alert = (
            db.query(Alert)
            .filter(
                Alert.id == alert_id,
                (Alert.user_external_id == x_user_id) | (Alert.user_email == app_user.email),
            )
            .first()
        )
        if not alert:
            raise HTTPException(status_code=404, detail="Alert not found")

        # Plan enforcement: active alert limit (activation)
        requested_is_active = getattr(payload, "is_active", None)
        if requested_is_active is True and alert.is_active is not True:
            limit = int(getattr(app_user, "plan_active_alert_limit", 1) or 1)

            active_count = (
                db.query(Alert)
                .filter(
                    ((Alert.user_external_id == x_user_id) | (Alert.user_email == app_user.email)),
                    Alert.is_active == True,  # noqa: E712
                )
                .count()
            )

            if active_count >= limit:
                raise HTTPException(status_code=403, detail={"code": "ALERT_LIMIT_REACHED"})

        # Plan enforcement: departure window limit (update)
        window_limit = int(getattr(app_user, "plan_max_departure_window_days", 15) or 15)

        effective_start = getattr(payload, "departure_start", None) or alert.departure_start
        effective_end = getattr(payload, "departure_end", None) or alert.departure_end

        if effective_start and effective_end:
            window_days = (effective_end - effective_start).days + 1
            if window_days > window_limit:
                raise HTTPException(status_code=403, detail={"code": "WINDOW_LIMIT_EXCEEDED"})

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

        pax = getattr(payload, "passengers", None)
        if pax is not None:
            max_passengers = get_config_int("MAX_PASSENGERS", 4)
            alert.passengers = min(max(1, int(pax)), max_passengers)

        alert.updated_at = datetime.utcnow()
        db.commit()

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
        if not x_user_id:
            raise HTTPException(status_code=401, detail={"code": "UNAUTHORIZED"})

        app_user = db.query(AppUser).filter(AppUser.external_id == x_user_id).first()
        if not app_user:
            raise HTTPException(status_code=403, detail="Unknown user")

        alert = (
            db.query(Alert)
            .filter(
                Alert.id == alert_id,
                (Alert.user_external_id == x_user_id) | (Alert.user_email == app_user.email),
            )
            .first()
        )
        if not alert:
            raise HTTPException(status_code=404, detail="Alert not found")

        db.query(AlertRun).filter(AlertRun.alert_id == alert.id).delete()
        db.delete(alert)
        db.commit()

        return {"status": "ok", "id": alert_id}
    finally:
        db.close()

# =====================================================================
# SECTION END: ALERT ROUTES
# =====================================================================
