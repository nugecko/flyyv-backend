"""schemas/alerts.py - Pydantic models for alert CRUD."""

from datetime import date, datetime
from typing import List, Optional

from pydantic import BaseModel


class AlertCreate(BaseModel):
    # Legacy field — not trusted for auth, kept for backward compatibility
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

    # Still returned for UI display. Ownership is via X-User-Id.
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


class AlertUpdatePayload(BaseModel):
    # Basic status
    is_active: Optional[bool] = None

    # Core alert rule
    alert_type: Optional[str] = None
    mode: Optional[str] = None
    max_price: Optional[int] = None

    # Date windows
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


class AlertWithStatsResponse(AlertResponse):
    best_price: Optional[int] = None
    last_run_at: Optional[datetime] = None
    last_notified_at: Optional[datetime] = None
    last_notified_price: Optional[int] = None
    created_at: datetime
    updated_at: datetime

    class Config:
        orm_mode = True
