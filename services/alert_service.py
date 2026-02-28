"""
services/alert_service.py

Alert engine:
- process_alert: runs one alert scan and sends email if criteria met
- build_search_params_for_alert: translates Alert DB model to SearchParams
- run_all_alerts_cycle: the cron entry point (called by run_alerts_cycle.py)
"""

import json
import traceback
from datetime import datetime, timedelta
from types import SimpleNamespace
from typing import Any, List, Optional
from uuid import uuid4

from sqlalchemy import func, text as sql_text
from sqlalchemy.orm import Session

from alerts_email import send_alert_email_for_alert, send_smart_alert_email
from config import master_alerts_enabled, should_send_alert
from db import SessionLocal
from models import Alert, AlertRun, AppUser
from providers.factory import run_provider_scan
from schemas.search import SearchParams


# =====================================================================
# SECTION: HELPERS
# =====================================================================

def _derive_alert_passengers(alert: Any) -> int:
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
    Plan enforcement cadence:
    1/day => 24h, 3/day => 8h
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
    1) AppUser.external_id == alert.user_external_id
    2) AppUser.email == alert.user_email (legacy fallback)
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

    pax = _derive_alert_passengers(alert)

    # ---- Derive fixed nights ----
    # Priority: alert.nights field, then return_start - dep_start
    nights = None
    raw_nights = getattr(alert, "nights", None)
    if raw_nights is not None:
        try:
            nights = int(raw_nights)
        except Exception:
            nights = None

    if nights is None and alert.return_start:
        try:
            nights = max(1, (alert.return_start - dep_start).days)
        except Exception:
            nights = None

    if nights and nights > 0:
        # FlyyvFlex fixed-night: pass nights explicitly, keep full dep window
        # The pair generator uses: last_dep = latest - nights, which gives correct pairs
        latest_valid_dep = dep_end  # keep original window
        min_stay = nights
        max_stay = nights
    else:
        # Fallback: variable stay
        latest_valid_dep = dep_end
        if alert.return_start and alert.return_end:
            min_stay = max(1, (alert.return_start - dep_start).days)
            max_stay = max(min_stay, (alert.return_end - dep_start).days)
        else:
            min_stay = 1
            max_stay = 21

    return SearchParams(
        origin=alert.origin,
        destination=alert.destination,
        earliestDeparture=dep_start,
        latestDeparture=latest_valid_dep,
        minStayDays=min_stay,
        maxStayDays=max_stay,
        nights=nights if nights and nights > 0 else None,
        maxPrice=None,
        cabin=alert.cabin or "BUSINESS",
        passengers=pax,
        stopsFilter=None,
        maxOffersPerPair=120,
        maxOffersTotal=1200,
    )


# =====================================================================
# SECTION: PROCESS SINGLE ALERT
# =====================================================================

def process_alert(alert: Alert, db: Session) -> None:
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

    # ---- User resolution ----
    user = _get_user_for_alert(db, alert)

    alert.last_run_at = now
    alert.updated_at = now

    alert_run_id = str(uuid4())

    if not user:
        db.add(AlertRun(id=alert_run_id, alert_id=alert.id, run_at=now, price_found=None, sent=False, reason="no_user_for_alert"))
        db.commit()
        return

    if not should_send_alert(db, user):
        db.add(AlertRun(id=alert_run_id, alert_id=alert.id, run_at=now, price_found=None, sent=False, reason="alerts_disabled"))
        db.commit()
        return

    # ---- Duplicate guard (short window) ----
    if getattr(alert, "last_checked_at", None) is not None:
        age_seconds = (now - alert.last_checked_at).total_seconds()
        if age_seconds < 300:
            print(f"[alerts] skip recent_check run_id={alert_run_id} alert_id={alert.id} age_seconds={int(age_seconds)}")
            return

    alert.last_checked_at = now
    alert.updated_at = now
    db.commit()

    # ---- Create run row up front (FK anchor for snapshots) ----
    run_row = AlertRun(id=alert_run_id, alert_id=str(alert.id), run_at=now, price_found=None, sent=False, reason="started")
    db.add(run_row)
    db.commit()

    # ---- Guard: skip and expire alerts with past departure dates ----
    dep_end = getattr(alert, "departure_end", None) or getattr(alert, "departure_start", None)
    if dep_end and dep_end < now.date():
        print(f"[alerts] skipping past-date alert run_id={alert_run_id} alert_id={alert.id} dep_end={dep_end}")
        alert.is_active = False
        alert.updated_at = now
        run_row.reason = "expired_past_dates"
        db.add(run_row)
        db.commit()
        return

    # ---- Build scan params ----
    params = build_search_params_for_alert(alert)

    if getattr(alert, "max_price", None) is not None:
        try:
            scan_params = params.model_copy(update={"maxPrice": None})
        except Exception:
            try:
                scan_params = params.copy(update={"maxPrice": None})
            except Exception:
                from copy import deepcopy
                scan_params = deepcopy(params)
                scan_params.maxPrice = None
    else:
        scan_params = params

    # ---- Cache read or scan ----
    cache_hit = False
    options = None

    if alert.cache_expires_at and alert.cache_expires_at > now and alert.cache_payload_json:
        try:
            cached_list = json.loads(alert.cache_payload_json) or []
            options = [SimpleNamespace(**d) for d in cached_list]
            cache_hit = True
        except Exception as e:
            print(f"[alerts] cache read failed run_id={alert_run_id} alert_id={alert.id}: {e}")
            options = None
            cache_hit = False

    if options is None:
        options = run_provider_scan(scan_params)

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

    if not options:
        run_row.reason = "no_results_scan_empty"
        db.add(run_row)
        db.commit()
        return

    # ---- Cheapest and stored best ----
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

    # ---- Decide whether to send ----
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

    # ---- Snapshot (when sending) ----
    if should_send:
        try:
            def _opt_to_dict_for_snapshot(opt):
                if opt is None:
                    return {}
                if hasattr(opt, "model_dump"):
                    return opt.model_dump()
                if hasattr(opt, "dict"):
                    return opt.dict()
                return dict(getattr(opt, "__dict__", None) or {})

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

            pax = max(1, int(getattr(params, "passengers", 1) or 1))
            raw_top_results = [_opt_to_dict_for_snapshot(o) for o in (options_sorted or [])[:5]]

            top_results = []
            for item in raw_top_results:
                if not isinstance(item, dict):
                    top_results.append(item)
                    continue

                price_per_pax = _to_float(_pick_first(item, ["price_per_pax", "pricePerPax", "per_pax_price", "price"]))
                total_price = _to_float(_pick_first(item, ["total_price", "totalPrice", "total"]))

                if price_per_pax is None and total_price is not None:
                    price_per_pax = round(total_price / pax, 2)
                if total_price is None and price_per_pax is not None:
                    total_price = round(price_per_pax * pax, 2)

                item["passengers"] = pax
                item["price_per_pax"] = price_per_pax
                item["total_price"] = total_price
                top_results.append(item)

            params_payload = {}
            for k in ["origin", "destination", "cabin", "passengers", "search_mode",
                      "earliestDeparture", "latestDeparture", "nights", "minStayDays",
                      "maxStayDays", "stopsFilter", "maxPrice", "currency"]:
                try:
                    v = getattr(params, k, None)
                    if hasattr(v, "isoformat"):
                        v = v.isoformat()
                    params_payload[k] = v
                except Exception:
                    pass

            cheapest_dict = top_results[0] if isinstance(top_results, list) and top_results else {}
            cheapest_currency = cheapest_dict.get("currency") if isinstance(cheapest_dict, dict) else None
            currency = cheapest_currency or params_payload.get("currency") or "GBP"

            best_price_val = None
            if isinstance(cheapest_dict, dict):
                best_price_val = _to_float(cheapest_dict.get("price_per_pax"))
            if best_price_val is None:
                best_price_val = _to_float(current_price)
            best_price_int = int(round(best_price_val)) if best_price_val is not None else None

            db.execute(
                sql_text("""
                    insert into alert_run_snapshots
                        (alert_run_id, alert_id, user_email, params, top_results, best_price_per_pax, currency, meta)
                    values
                        (:alert_run_id, :alert_id, :user_email,
                         CAST(:params AS jsonb), CAST(:top_results AS jsonb),
                         :best_price, :currency, CAST(:meta AS jsonb))
                    on conflict (alert_run_id) do nothing
                """),
                {
                    "alert_run_id": alert_run_id,
                    "alert_id": str(alert.id),
                    "user_email": getattr(alert, "user_email", None),
                    "params": json.dumps(params_payload),
                    "top_results": json.dumps(top_results),
                    "best_price": best_price_int,
                    "currency": str(currency),
                    "meta": json.dumps({
                        "cache_hit": bool(cache_hit),
                        "options_count": int(len(options_sorted or [])),
                        "top_results_count": int(len(top_results)),
                        "send_reason": str(send_reason),
                        "created_at_utc": now.isoformat(),
                    }),
                },
            )
            db.commit()

            print(
                f"[alerts] snapshot saved "
                f"run_id={alert_run_id} alert_id={alert.id} "
                f"best_price_per_pax={best_price_int} currency={currency}"
            )
        except Exception as e:
            print(f"[alerts] snapshot insert failed run_id={alert_run_id} alert_id={alert.id}: {e}")
            try:
                db.rollback()
            except Exception:
                pass

    # ---- Send email ----
    sent_flag = False

    if should_send:
        try:
            if getattr(alert, "mode", None) == "smart":
                send_smart_alert_email(alert, options_sorted, params, alert_run_id=alert_run_id)
            else:
                send_alert_email_for_alert(alert, cheapest, params, alert_run_id=alert_run_id)
            sent_flag = True
        except Exception as e:
            print(f"[alerts] Failed to send email run_id={alert_run_id} alert_id={alert.id}: {e}")
            sent_flag = False
            send_reason = "email_failed"

    # ---- Finalise run row and update alert ----
    try:
        run_row.price_found = current_price
        run_row.sent = bool(sent_flag)
        run_row.reason = str(send_reason)
        db.add(run_row)
    except Exception:
        db.query(AlertRun).filter(AlertRun.id == alert_run_id).update({
            "price_found": current_price,
            "sent": bool(sent_flag),
            "reason": str(send_reason),
        })

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


# =====================================================================
# SECTION: ALERTS CYCLE (CRON ENTRY POINT)
# =====================================================================

def run_all_alerts_cycle() -> None:
    if not master_alerts_enabled():
        print("[alerts] ALERTS_ENABLED is false, skipping alerts cycle")
        return

    from config import SMTP_USERNAME, SMTP_PASSWORD, ALERT_FROM_EMAIL
    if not (SMTP_USERNAME and SMTP_PASSWORD and ALERT_FROM_EMAIL):
        print("[alerts] SMTP not fully configured, skipping alerts cycle")
        return

    db = SessionLocal()
    try:
        from config import alerts_globally_enabled
        if not alerts_globally_enabled(db):
            print("[alerts] Global alerts disabled in admin_config, skipping alerts cycle")
            return

        today = datetime.utcnow().date()
        now = datetime.utcnow()

        # Expire alerts past departure window
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

                if not user:
                    eligible_alerts.append(a)
                    continue

                plan_tier = (getattr(user, "plan_tier", None) or "").lower()

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
                eligible_alerts.append(a)

        print(
            f"[alerts] Running alerts cycle for {len(eligible_alerts)} eligible alerts "
            f"(skipped {skipped} due to plan cadence)"
        )

        for alert in eligible_alerts:
            try:
                process_alert(alert, db)
            except Exception as e:
                print(f"[alerts] Error processing alert {getattr(alert, 'id', '?')}: {e}")
                traceback.print_exc()
                try:
                    db.rollback()
                except Exception:
                    pass

    finally:
        db.close()