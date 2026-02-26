"""routers/users.py - User sync, profile, and public config endpoints."""

from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Header, HTTPException, Request

from config import ALLOWED_TIERS, PLAN_DEFAULTS, get_config_int
from db import SessionLocal
from models import Alert, AppUser
from schemas.users import (
    ProfileAlertUsage,
    ProfileEntitlements,
    ProfileResponse,
    ProfileUser,
    PublicConfig,
    SubscriptionInfo,
    UserSyncPayload,
    WalletInfo,
)
from services.search_service import USER_WALLETS
from sqlalchemy import func

router = APIRouter()


@router.get("/public-config", response_model=PublicConfig)
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


@router.post("/user-sync")
def user_sync(payload: UserSyncPayload):
    print(f"[user-sync] payload: email={payload.email}, plan_tier_code={payload.plan_tier_code}, external_id={getattr(payload, 'external_id', None) or getattr(payload, 'id', None)}")
    db = SessionLocal()
    canonical_external_id = None

    try:
        canonical_external_id = (
            (getattr(payload, "external_id", None) or "").strip()
            or (getattr(payload, "id", None) or "").strip()
            or (getattr(payload, "user_id", None) or "").strip()
        )
        if not canonical_external_id:
            raise HTTPException(status_code=400, detail="Missing external_id")

        FREE_DEFAULTS = PLAN_DEFAULTS["free"]

        # Primary lookup: external id
        user = db.query(AppUser).filter(AppUser.external_id == canonical_external_id).first()

        # Secondary lookup: same email, new external_id
        if user is None and payload.email:
            email_norm = payload.email.strip().lower()
            user = db.query(AppUser).filter(func.lower(AppUser.email) == email_norm).first()
            if user is not None:
                user.external_id = canonical_external_id

        if user is None:
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

        # Ensure entitlements exist
        if getattr(user, "plan_tier", None) in (None, ""):
            user.plan_tier = FREE_DEFAULTS["plan_tier"]

        if getattr(user, "plan_active_alert_limit", None) is None:
            user.plan_active_alert_limit = FREE_DEFAULTS["plan_active_alert_limit"]

        if getattr(user, "plan_max_departure_window_days", None) is None:
            user.plan_max_departure_window_days = FREE_DEFAULTS["plan_max_departure_window_days"]

        if getattr(user, "plan_checks_per_day", None) is None:
            user.plan_checks_per_day = FREE_DEFAULTS["plan_checks_per_day"]

        # Lock plan values to tier defaults
        if user.plan_tier in PLAN_DEFAULTS:
            d = PLAN_DEFAULTS[user.plan_tier]
            user.plan_active_alert_limit = d["plan_active_alert_limit"]
            user.plan_max_departure_window_days = d["plan_max_departure_window_days"]
            user.plan_checks_per_day = d["plan_checks_per_day"]

        print(f"[user-sync] ABOUT TO COMMIT - tier={user.plan_tier}, limit={user.plan_active_alert_limit}")
        try:
            db.commit()
            print(f"[user-sync] COMMITTED - tier is now {user.plan_tier}")
        except Exception as commit_error:
            import traceback
            print(f"[user-sync] COMMIT FAILED: {commit_error}")
            print(f"[user-sync] COMMIT TRACEBACK: {traceback.format_exc()}")
            raise
        db.refresh(user)
        print(f"[user-sync] AFTER REFRESH - tier is {user.plan_tier}")

        db.commit()
        db.refresh(user)
        return {"status": "ok", "id": user.id}

    except Exception:
        db.rollback()

        if canonical_external_id:
            existing = db.query(AppUser).filter(AppUser.external_id == canonical_external_id).first()
            if existing is not None:
                return {"status": "ok", "id": existing.id}

        if payload.email:
            email_norm = payload.email.strip().lower()
            existing = db.query(AppUser).filter(func.lower(AppUser.email) == email_norm).first()
            if existing is not None:
                return {"status": "ok", "id": existing.id}

        raise

    finally:
        db.close()


@router.post("/base44/user-webhook")
async def base44_user_webhook(
    request: Request,
    x_webhook_secret: str = Header(None, alias="X-Webhook-Secret"),
):
    from config import BASE44_WEBHOOK_SECRET
    expected = BASE44_WEBHOOK_SECRET
    if expected and x_webhook_secret != expected:
        raise HTTPException(status_code=401, detail="Invalid webhook secret")

    payload = await request.json()

    user_obj = None
    if isinstance(payload, dict):
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


@router.get("/profile", response_model=ProfileResponse)
def get_profile(x_user_id: str = Header(..., alias="X-User-Id")):
    wallet_balance = USER_WALLETS.get(x_user_id, 0)

    app_user = None
    active_alerts = 0
    display_name = "Member"
    external_id = x_user_id
    joined_at = None

    db = SessionLocal()
    try:
        app_user = db.query(AppUser).filter(AppUser.external_id == x_user_id).first()

        if app_user:
            external_id = app_user.external_id
            joined_at = app_user.created_at

            active_alerts = (
                db.query(Alert)
                .filter(
                    (Alert.user_external_id == app_user.external_id) | (Alert.user_email == app_user.email),
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
