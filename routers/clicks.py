"""routers/clicks.py - Click tracking and redirect endpoints."""

from typing import Optional

from fastapi import APIRouter, Header, HTTPException
from fastapi.responses import RedirectResponse

from schemas.clicks import ClickRegisterRequest, ClickRegisterResponse
from services.click_service import follow_click, register_click

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
