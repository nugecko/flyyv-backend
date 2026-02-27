"""
services/click_service.py

Click tracking / redirect system.

Flow:
  1. Base44 frontend calls POST /click with offer details + destination_url
  2. Backend writes OfferClick row to Postgres, returns click_id
  3. Backend also caches click_id -> url in CLICK_REDIRECTS (in-memory)
  4. Frontend opens api.flyyv.com/go/{click_id}
  5. Backend reads CLICK_REDIRECTS (or falls back to DB), returns 302

Why a redirect layer?
  Flight API (and other providers) return booking handoff URLs that we
  cannot instrument with tracking params without risking breaking the
  redirect chain. This approach keeps the URL untouched while logging
  every click server-side.

Analytics:
  All OfferClick rows are available in Directus for querying.
  Fields allow segmenting by: airline, route, price range, source,
  cabin, alert vs search, date window, provider.
"""

from datetime import date, datetime
from typing import Dict, Optional
from uuid import uuid4

from db import SessionLocal
from models import OfferClick
from schemas.clicks import ClickRegisterRequest, ClickRegisterResponse
from config import FRONTEND_BASE_URL


# In-memory cache: click_id -> destination_url
# Fallback to DB if missing (e.g. after a container restart).
CLICK_REDIRECTS: Dict[str, str] = {}


def register_click(
    payload: ClickRegisterRequest,
    user_external_id: Optional[str] = None,
) -> ClickRegisterResponse:
    """
    Write click to Postgres, cache redirect, return click_id.
    Called by POST /click.
    """
    click_id = str(uuid4())

    # Parse dates defensively
    dep_date = None
    ret_date = None
    try:
        if payload.departure_date:
            dep_date = date.fromisoformat(payload.departure_date)
    except Exception:
        pass
    try:
        if payload.return_date:
            ret_date = date.fromisoformat(payload.return_date)
    except Exception:
        pass

    # Cache in memory for fast redirect (survives until container restart)
    CLICK_REDIRECTS[click_id] = payload.destination_url

    # Persist to DB (for analytics and restart recovery)
    db = SessionLocal()
    try:
        click = OfferClick(
            id=click_id,
            clicked_at=datetime.utcnow(),
            user_external_id=user_external_id,
            offer_id=payload.offer_id,
            airline_code=payload.airline_code,
            price=payload.price,
            currency=payload.currency,
            origin=payload.origin,
            destination=payload.destination,
            departure_date=dep_date,
            return_date=ret_date,
            cabin=payload.cabin,
            stops=payload.stops,
            source=payload.source or "search",
            job_id=payload.job_id,
            alert_id=payload.alert_id,
            destination_url=payload.destination_url,
            provider=payload.provider or "ttn",
            redirect_followed=False,  # will be updated when /go/{click_id} is hit
        )
        db.add(click)
        db.commit()
    except Exception as e:
        print(f"[click] DB write failed click_id={click_id}: {e}")
        try:
            db.rollback()
        except Exception:
            pass
    finally:
        db.close()

    redirect_url = f"{FRONTEND_BASE_URL.rstrip('/')}/go/{click_id}"

    return ClickRegisterResponse(click_id=click_id, redirect_url=redirect_url)


def follow_click(click_id: str) -> Optional[str]:
    """
    Look up the destination URL for a click_id.
    1. Check in-memory cache (fast)
    2. Fall back to DB (handles container restarts)
    Returns the destination URL, or None if not found.
    """
    # Fast path: in-memory
    url = CLICK_REDIRECTS.get(click_id)
    if url:
        _mark_followed(click_id)
        return url

    # DB fallback
    db = SessionLocal()
    try:
        click = db.query(OfferClick).filter(OfferClick.id == click_id).first()
        if click and click.destination_url:
            url = click.destination_url
            CLICK_REDIRECTS[click_id] = url  # re-cache
            try:
                click.redirect_followed = True
                db.commit()
            except Exception:
                pass
            return url
    except Exception as e:
        print(f"[click] DB fallback failed click_id={click_id}: {e}")
    finally:
        db.close()

    return None


def track_email_click(
    destination_url: str,
    src: str = "email",
    alert_id: Optional[str] = None,
    run_id: Optional[str] = None,
    airline_code: Optional[str] = None,
    origin: Optional[str] = None,
    destination_iata: Optional[str] = None,
    departure_date: Optional[str] = None,
    return_date: Optional[str] = None,
    cabin: Optional[str] = None,
    passengers: Optional[int] = None,
    price: Optional[float] = None,
) -> None:
    """
    Best-effort click log for email tracking links (GET /go).
    Writes to OfferClick table. Never raises — email redirect must always succeed.
    """
    click_id = str(uuid4())

    dep_date = None
    ret_date = None
    try:
        if departure_date:
            dep_date = date.fromisoformat(departure_date)
    except Exception:
        pass
    try:
        if return_date:
            ret_date = date.fromisoformat(return_date)
    except Exception:
        pass

    db = SessionLocal()
    try:
        click = OfferClick(
            id=click_id,
            clicked_at=datetime.utcnow(),
            user_external_id=None,
            offer_id=run_id,          # reuse offer_id field for run_id context
            airline_code=airline_code,
            price=price,
            currency="GBP",
            origin=origin,
            destination=destination_iata,
            departure_date=dep_date,
            return_date=ret_date,
            cabin=cabin,
            stops=None,
            source=src,
            job_id=None,
            alert_id=alert_id,
            destination_url=destination_url,
            provider=src or "email",
            redirect_followed=True,
        )
        db.add(click)
        db.commit()
    except Exception as e:
        print(f"[click] email track write failed click_id={click_id}: {e}")
        try:
            db.rollback()
        except Exception:
            pass
    finally:
        db.close()



    """Best-effort: mark redirect_followed=True in DB."""
    db = SessionLocal()
    try:
        db.query(OfferClick).filter(OfferClick.id == click_id).update({"redirect_followed": True})
        db.commit()
    except Exception:
        pass
    finally:
        db.close()