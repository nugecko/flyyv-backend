"""schemas/users.py - Pydantic models for user sync and profile endpoints."""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel


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
