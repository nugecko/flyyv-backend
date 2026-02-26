"""routers/alerts.py - Alert CRUD: create, list, update, delete."""

from datetime import datetime
from typing import List, Optional
from uuid import uuid4

from fastapi import APIRouter, Header, HTTPException, Request

from alerts_email import send_alert_confirmation_email
from config import get_config_int
from db import SessionLocal
from models import Alert, AlertRun, AppUser
from schemas.alerts import AlertCreate, AlertOut, AlertUpdatePayload
from services.alert_service import _derive_alert_passengers
from sqlalchemy import func

router = APIRouter()


def _alert_state(a: Alert) -> str:
    today = datetime.utcnow().date()
    end = a.departure_end or a.departure_start
    if end and today > end:
        return "expired"
    return "active" if a.is_active else "paused"


@router.post("/alerts", response_model=AlertOut)
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

        app_user = db.query(AppUser).filter(AppUser.external_id == x_user_id).first()
        if not app_user:
            raise HTTPException(status_code=401, detail={"code": "UNAUTHORIZED"})

        active_alerts = (
            db.query(Alert)
            .filter(
                (Alert.user_external_id == x_user_id) | (Alert.user_email == app_user.email),
                Alert.is_active == True,  # noqa: E712
            )
            .count()
        )

        limit = int(getattr(app_user, "plan_active_alert_limit", 1) or 1)

        if active_alerts >= limit:
            raise HTTPException(status_code=403, detail={"code": "ALERT_LIMIT_REACHED"})

        # Plan enforcement: departure window limit
        window_limit = int(getattr(app_user, "plan_max_departure_window_days", 15) or 15)
        if payload.departure_start and payload.departure_end:
            window_days = (payload.departure_end - payload.departure_start).days + 1
            if window_days > window_limit:
                raise HTTPException(status_code=403, detail={"code": "WINDOW_LIMIT_EXCEEDED"})

        if pax > max_passengers:
            pax = max_passengers

        dep_start = payload.departure_start
        dep_end = payload.departure_end or dep_start

        resolved_alert_type = (
            "under_price" if payload.max_price is not None
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


@router.get("/alerts", response_model=List[AlertOut])
def get_alerts(
    request: Request,
    x_user_id: Optional[str] = Header(None, alias="X-User-Id"),
    include_inactive: bool = False,
):
    db = SessionLocal()
    try:
        if "email" in request.query_params:
            raise HTTPException(status_code=400, detail="Email query param is not supported, use X-User-Id header")

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
            result.append(AlertOut(
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
            ))

        return result
    finally:
        db.close()


@router.patch("/alerts/{alert_id}")
def update_alert(
    alert_id: str,
    payload: AlertUpdatePayload,
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

        # Plan enforcement: active alert limit (activation)
        requested_is_active = getattr(payload, "is_active", None)
        if requested_is_active is True and alert.is_active is not True:
            limit = int(getattr(app_user, "plan_active_alert_limit", 1) or 1)
            active_count = (
                db.query(Alert)
                .filter(
                    (Alert.user_external_id == x_user_id) | (Alert.user_email == app_user.email),
                    Alert.is_active == True,  # noqa: E712
                )
                .count()
            )
            if active_count >= limit:
                raise HTTPException(status_code=403, detail={"code": "ALERT_LIMIT_REACHED"})

        # Plan enforcement: departure window limit
        window_limit = int(getattr(app_user, "plan_max_departure_window_days", 15) or 15)
        effective_start = getattr(payload, "departure_start", None) or alert.departure_start
        effective_end = getattr(payload, "departure_end", None) or alert.departure_end
        if effective_start and effective_end:
            window_days = (effective_end - effective_start).days + 1
            if window_days > window_limit:
                raise HTTPException(status_code=403, detail={"code": "WINDOW_LIMIT_EXCEEDED"})

        for field in ("alert_type", "max_price", "departure_start", "departure_end",
                      "return_start", "return_end", "mode", "is_active"):
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


@router.delete("/alerts/{alert_id}")
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
