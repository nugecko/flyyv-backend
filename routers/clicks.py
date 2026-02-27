"""routers/clicks.py - Click tracking and redirect endpoints."""

from typing import Optional

from fastapi import APIRouter, Header, HTTPException, Query
from fastapi.responses import RedirectResponse

from schemas.clicks import ClickRegisterRequest, ClickRegisterResponse
from services.click_service import follow_click, register_click, track_email_click

router = APIRouter()


@router.post("/click", response_model=ClickRegisterResponse)
def register_click_endpoint(
    payload: ClickRegisterRequest,
    x_user_id: Optional[str] = Header(None, alias="X-User-Id"),
):
    """
    Called by Base44 frontend immediately before opening a booking URL.

    1. Validates destination_url is present.
    2. Writes OfferClick row to Postgres.
    3. Returns click_id which can be used to construct a redirect URL.

    The frontend should open:
      api.flyyv.com/go/{click_id}
    which will 302 to the actual booking URL.
    """
    if not payload.destination_url:
        raise HTTPException(status_code=400, detail="destination_url is required")

    result = register_click(payload, user_external_id=x_user_id)
    return result


@router.get("/go")
def email_click_redirect(
    url: str = Query(..., description="Destination URL to redirect to"),
    src: Optional[str] = Query(None, description="Source: skyscanner|kayak|airline|flyyv"),
    alert_id: Optional[str] = Query(None),
    run_id: Optional[str] = Query(None),
    airline: Optional[str] = Query(None),
    origin: Optional[str] = Query(None),
    destination: Optional[str] = Query(None),
    dep: Optional[str] = Query(None),
    ret: Optional[str] = Query(None),
    cabin: Optional[str] = Query(None),
    pax: Optional[int] = Query(None),
    price: Optional[float] = Query(None),
):
    """
    Stateless tracking redirect for email links.
    Logs click to OfferClick table then immediately 302s to the destination.
    Used by alert emails — no JS or frontend API call required.

    Example:
      GET /go?url=https://skyscanner.com/...&src=skyscanner&alert_id=abc&airline=BA
    """
    if not url:
        raise HTTPException(status_code=400, detail="url is required")

    track_email_click(
        destination_url=url,
        src=src or "email",
        alert_id=alert_id,
        run_id=run_id,
        airline_code=airline,
        origin=origin,
        destination_iata=destination,
        departure_date=dep,
        return_date=ret,
        cabin=cabin,
        passengers=pax,
        price=price,
    )

    return RedirectResponse(url=url, status_code=302)


@router.get("/go/{click_id}")
def follow_click_redirect(click_id: str):
    """
    Redirect the user to the booking URL associated with a click_id.
    Looks up in-memory cache first, then falls back to DB.
    """
    url = follow_click(click_id)

    if not url:
        raise HTTPException(
            status_code=404,
            detail="Click not found. The link may have expired or the server restarted.",
        )

    return RedirectResponse(url=url, status_code=302)
