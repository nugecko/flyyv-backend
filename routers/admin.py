"""routers/admin.py - Admin endpoints, health checks, debug routes, alert triggers."""

from datetime import datetime, timedelta
from typing import Optional
from uuid import UUID

from fastapi import APIRouter, BackgroundTasks, Header, HTTPException
from pydantic import BaseModel
from sqlalchemy import func, text

from config import ADMIN_API_TOKEN, PLAN_DEFAULTS, get_config_bool, get_config_int
from db import SessionLocal
from models import AppUser
from providers.duffel import duffel_get
from schemas.users import AdminConfigUpdatePayload
from services.alert_service import run_all_alerts_cycle
from services.search_service import USER_WALLETS

router = APIRouter()


# =====================================================================
# SECTION: PYDANTIC MODELS (admin-specific)
# =====================================================================

class CreditUpdateRequest(BaseModel):
    userId: str
    delta: Optional[int] = None
    amount: Optional[int] = None
    creditAmount: Optional[int] = None
    value: Optional[int] = None


class AdminSyncRequest(BaseModel):
    email: Optional[str] = None
    external_id: Optional[str] = None
    plan_tier_code: str


# =====================================================================
# SECTION: HEALTH AND DEBUG ROUTES
# =====================================================================

@router.get("/")
def home():
    return {"message": "Flyyv backend is running"}


@router.get("/health")
def health():
    return {"status": "ok"}


@router.get("/debug-duffel-get")
def debug_duffel_get():
    res = duffel_get("/air/airlines", params={"limit": 1})
    return {"ok": True, "type": str(type(res)), "sample": res}


@router.get("/routes")
def list_routes_handler():
    # Imported lazily to avoid circular import
    from main import app
    return [route.path for route in app.routes]


@router.get("/test-email-alert")
def test_email_alert():
    from main import _smtp_send_test
    _smtp_send_test()
    return {"detail": "Test alert email sent"}


@router.get("/test-email-confirmation")
def test_email_confirmation(email: Optional[str] = None, flex: int = 0):
    from config import ALERT_TO_EMAIL
    from alerts_email import send_alert_confirmation_email

    to = email or ALERT_TO_EMAIL

    class DummyAlert:
        user_email = to
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
    return {"detail": f"Test confirmation email sent to {to}, flex={flex}"}


@router.get("/test-email-smart-alert")
def test_email_smart_alert(email: Optional[str] = None):
    from config import ALERT_TO_EMAIL
    from alerts_email import send_smart_alert_email

    to = email or ALERT_TO_EMAIL
    print(f"[test-email-smart-alert] START to={to}")

    class DummyAlert:
        user_email = to
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
        DummyOption((datetime.utcnow() + timedelta(days=5)).date().isoformat(), (datetime.utcnow() + timedelta(days=12)).date().isoformat(), 1890, "British Airways"),
        DummyOption((datetime.utcnow() + timedelta(days=8)).date().isoformat(), (datetime.utcnow() + timedelta(days=15)).date().isoformat(), 2010, "EL AL"),
        DummyOption((datetime.utcnow() + timedelta(days=11)).date().isoformat(), (datetime.utcnow() + timedelta(days=18)).date().isoformat(), 2140, "Lufthansa"),
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
    print(f"[test-email-smart-alert] SENT OK to={to}")
    return {"detail": f"Test smart alert email sent to {to}"}


@router.get("/trigger-daily-alert")
def trigger_daily_alert(background_tasks: BackgroundTasks):
    from config import ALERTS_ENABLED
    if not ALERTS_ENABLED:
        return {"detail": "Alerts are currently disabled via environment"}

    system_enabled = get_config_bool("ALERTS_SYSTEM_ENABLED", True)
    if not system_enabled:
        return {"detail": "Alerts are currently disabled in admin config"}

    background_tasks.add_task(run_all_alerts_cycle)
    return {"detail": "Alerts cycle queued"}


# =====================================================================
# SECTION: ALERT RUN SNAPSHOT
# =====================================================================

@router.get("/alert-run-snapshot/{alert_run_id}")
def get_alert_run_snapshot(
    alert_run_id: str,
    x_user_id: Optional[str] = Header(None, alias="X-User-Id"),
    x_admin_token: Optional[str] = Header(None, alias="X-Admin-Token"),
):
    try:
        run_uuid = str(UUID(alert_run_id))
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid alert_run_id, must be a UUID")

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
            JOIN alerts a ON a.id = ar.alert_id
            JOIN alert_run_snapshots ars ON ars.alert_run_id = ar.id
            WHERE ar.id = :rid
        """

        bind = {"rid": run_uuid}

        if not is_admin:
            sql += " AND a.user_external_id = :uid"
            bind["uid"] = x_user_id

        sql += " ORDER BY ars.created_at DESC LIMIT 1"

        row = db.execute(text(sql), bind).mappings().first()

        if not row:
            raise HTTPException(status_code=404, detail="Snapshot not found for this alertRunId")

        params = row["params"] or {}
        meta = row["meta"] or {}
        top_results = row["top_results"] or []

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
            if "price_per_pax" in item and item["price_per_pax"] is not None:
                return item
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
# SECTION: ADMIN CREDITS
# =====================================================================

@router.post("/admin/add-credits")
def admin_add_credits(
    payload: CreditUpdateRequest,
    x_admin_token: Optional[str] = Header(None, alias="X-Admin-Token"),
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
        payload.delta if payload.delta is not None
        else payload.amount if payload.amount is not None
        else payload.creditAmount if payload.creditAmount is not None
        else payload.value
    )

    if change_amount is None:
        raise HTTPException(status_code=400, detail="Missing credit amount. Expected one of: amount, delta, creditAmount, value.")

    current = USER_WALLETS.get(payload.userId, 0)
    new_balance = max(0, current + change_amount)
    USER_WALLETS[payload.userId] = new_balance

    return {"userId": payload.userId, "newBalance": new_balance}


# =====================================================================
# SECTION: ADMIN FORCE SYNC USER TIER
# =====================================================================

@router.post("/admin/sync-user-tier")
def admin_sync_user_tier(
    payload: AdminSyncRequest,
    x_user_id: Optional[str] = Header(None, alias="X-User-Id"),
):
    db = SessionLocal()
    try:
        if not x_user_id:
            raise HTTPException(status_code=401, detail="X-User-Id header required")

        admin_user = db.query(AppUser).filter(AppUser.external_id == x_user_id).first()
        if not admin_user or admin_user.plan_tier != "admin":
            raise HTTPException(status_code=403, detail="Admin access required")

        if not payload.email and not payload.external_id:
            raise HTTPException(status_code=400, detail="Must provide email or external_id")

        if payload.external_id:
            target_user = db.query(AppUser).filter(AppUser.external_id == payload.external_id).first()
        else:
            target_user = db.query(AppUser).filter(func.lower(AppUser.email) == payload.email.lower()).first()

        if not target_user:
            raise HTTPException(status_code=404, detail="User not found")

        tier_norm = payload.plan_tier_code.strip().lower()
        if tier_norm not in PLAN_DEFAULTS:
            raise HTTPException(status_code=400, detail=f"Invalid tier: {tier_norm}. Must be one of: {list(PLAN_DEFAULTS.keys())}")

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
