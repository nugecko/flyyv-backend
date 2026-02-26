"""schemas/clicks.py - Pydantic models for the click tracking / redirect system."""

from datetime import date
from typing import Optional

from pydantic import BaseModel


class ClickRegisterRequest(BaseModel):
    """
    Sent by Base44 frontend immediately before opening a booking URL.
    Returns a click_id that can be used with GET /go/{click_id} to redirect.
    """
    destination_url: str

    offer_id: Optional[str] = None
    airline_code: Optional[str] = None
    price: Optional[float] = None
    currency: Optional[str] = None
    origin: Optional[str] = None
    destination: Optional[str] = None
    departure_date: Optional[str] = None   # YYYY-MM-DD string
    return_date: Optional[str] = None      # YYYY-MM-DD string
    cabin: Optional[str] = None
    stops: Optional[int] = None

    # "search" | "alert_email" | "preview"
    source: Optional[str] = "search"
    job_id: Optional[str] = None
    alert_id: Optional[str] = None

    provider: Optional[str] = "ttn"


class ClickRegisterResponse(BaseModel):
    click_id: str
    redirect_url: str
