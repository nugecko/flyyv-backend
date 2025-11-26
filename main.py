import os
import re
from datetime import date, timedelta, datetime
from typing import List, Optional, Tuple

import requests
from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from airlines import AIRLINE_NAMES, AIRLINE_BOOKING_URLS


# ------------- Models ------------- #

class SearchParams(BaseModel):
    origin: str
    destination: str
    earliestDeparture: date
    latestDeparture: date
    minStayDays: int
    maxStayDays: int
    # None means no price limit
    maxPrice: Optional[float] = None
    cabin: str = "BUSINESS"
    passengers: int = 1
    # Optional filter for number of stops, for example [0] or [0,1,2]
    # 3 is treated as "3 or more stops"
    stopsFilter: Optional[List[int]] = None


class FlightOption(BaseModel):
    id: str
    airline: str
    airlineCode: Optional[str] = None
    price: float
    currency: str
    departureDate: str
    returnDate: str
    stops: int

    durationMinutes: int
    totalDurationMinutes: Optional[int] = None
    duration: Optional[str] = None

    bookingUrl: Optional[str] = None
    url: Optional[str] = None


class CreditUpdateRequest(BaseModel):
    userId: str
    amount: Optional[int] = None
    delta: Optional[int] = None
    creditAmount: Optional[int] = None
    value: Optional[int] = None
    reason: Optional[str] = None


# ------------- FastAPI app ------------- #

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # later you can restrict this to your Base44 domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ------------- Admin token ------------- #

ADMIN_API_TOKEN = os.getenv("ADMIN_API_TOKEN")


# ------------- Duffel config (HTTP, no SDK) ------------- #

DUFFEL_ACCESS_TOKEN = os.getenv("DUFFEL_ACCESS_TOKEN")
DUFFEL_API_URL = "https://api.duffel.com/air"
DUFFEL_API_VERSION = "v2"   # important: v1 is now deprecated


def duffel_headers() -> dict:
    if not DUFFEL_ACCESS_TOKEN:
        raise RuntimeError("DUFFEL_ACCESS_TOKEN is not configured")
    return {
        "Authorization": f"Bearer {DUFFEL_ACCESS_TOKEN}",
        "Duffel-Version": DUFFEL_API_VERSION,
        "Accept": "application/json",
        "Content-Type": "application/json",
        "Accept-Encoding": "gzip",
    }


# ------------- Helpers ------------- #

def parse_iso_duration_to_minutes(iso_duration: str) -> int:
    if not iso_duration:
        return 0
    pattern = r"P(?:(\d+)D)?T(?:(\d+)H)?(?:(\d+)M)?"
    match = re.match(pattern, iso_duration)
    if not match:
        return 0
    days_str, hours_str, minutes_str = match.groups()
    days = int(days_str) if days_str else 0
    hours = int(hours_str) if hours_str else 0
    minutes = int(minutes_str) if minutes_str else 0
    return (days * 24 + hours) * 60 + minutes


def generate_date_pairs(params: SearchParams, max_pairs: int = 60) -> List[Tuple[date, date]]:
    """
    Generate (departure, return) pairs for all valid dates
    inside the window, respecting minStayDays and maxStayDays.
    """
    pairs: List[Tuple[date, date]] = []

    # Normalise stay range
    min_stay = max(1, params.minStayDays)
    max_stay = max(min_stay, params.maxStayDays)

    stays = list(range(min_stay, max_stay + 1))

    current = params.earliestDeparture

    while current <= params.latestDeparture and len(pairs) < max_pairs:
        for stay in stays:
            ret = current + timedelta(days=stay)
            if ret <= params.latestDeparture:
                pairs.append((current, ret))
                if len(pairs) >= max_pairs:
                    break
        current += timedelta(days=1)

    return pairs


def map_duffel_offer_to_option(offer: dict, dep: date, ret: date, index: int) -> FlightOption:
    """
    Map a Duffel offer (raw JSON) to your FlightOption model.
    """
    price = float(offer["total_amount"])
    currency = offer["total_currency"]

    slices = offer.get("slices", [])
    outbound = slices[0] if slices else {}
    segments = outbound.get("segments", [])
    stops_outbound = max(0, len(segments) - 1)

    # Try to compute duration from the first to last segment on the outbound slice
    duration_minutes = 0
    if segments:
        try:
            first_seg = segments[0]
            last_seg = segments[-1]
            dep_dt = datetime.fromisoformat(first_seg["departing_at"].replace("Z", "+00:00"))
            arr_dt = datetime.fromisoformat(last_seg["arriving_at"].replace("Z", "+00:00"))
            duration_minutes = int((arr_dt - dep_dt).total_seconds() // 60)
        except Exception:
            duration_minutes = 0

    owner = offer.get("owner", {})
    airline_code = owner.get("iata_code") or ""
    airline_name = AIRLINE_NAMES.get(airline_code, owner.get("name", airline_code or "Airline"))
    booking_url = AIRLINE_BOOKING_URLS.get(airline_code)

    return FlightOption(
        id=offer["id"],
        airline=airline_name,
        airlineCode=airline_code or None,
        price=price,
        currency=currency,
        departureDate=dep.isoformat(),
        returnDate=ret.isoformat(),
        stops=stops_outbound,
        durationMinutes=duration_minutes,
        totalDurationMinutes=duration_minutes,
        duration=None,
        bookingUrl=booking_url,
        url=booking_url,
    )


def apply_filters(options: List[FlightOption], params: SearchParams) -> List[FlightOption]:
    """
    Apply price and stops filters, then sort by price.
    """
    filtered = list(options)

    # Price filter
    if params.maxPrice is not None and params.maxPrice > 0:
        filtered = [o for o in filtered if o.price <= params.maxPrice]

    # Stops filter, values like [0], [0, 1], [0, 1, 2], [2, 3] and so on
    if params.stopsFilter:
        allowed = set(params.stopsFilter)
        if 3 in allowed:
            # 3 means "3 or more stops"
            filtered = [o for o in filtered if (o.stops in allowed or o.stops >= 3)]
        else:
            filtered = [o for o in filtered if o.stops in allowed]

    filtered.sort(key=lambda x: x.price)
    return filtered


# ------------- Routes: health and search ------------- #

@app.get("/")
def home():
    return {"message": "Flyyv backend is running"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/search-business")
def search_business(params: SearchParams):
    """
    Main endpoint used by the Base44 frontend.

    Behaviour:
    - Generate all valid (departure, return) date pairs inside the window.
    - For each pair, call Duffel for a round trip (two slices).
    - Map to FlightOption, then apply price and stops filters.

    No bookings are created, this is search only.
    """

    if not DUFFEL_ACCESS_TOKEN:
        return {
            "status": "error",
            "source": "duffel_missing_token",
            "message": "Duffel access token is not configured",
            "options": [],
        }

    headers = duffel_headers()
    offers_collected: List[FlightOption] = []

    try:
        date_pairs = generate_date_pairs(params, max_pairs=60)

        # Safety fallback: if no pairs, still try once
        if not date_pairs:
            date_pairs = [(params.earliestDeparture, params.latestDeparture)]

        for dep, ret in date_pairs:
            slices = [
                {
                    "origin": params.origin,
                    "destination": params.destination,
                    "departure_date": dep.isoformat(),
                },
                {
                    "origin": params.destination,
                    "destination": params.origin,
                    "departure_date": ret.isoformat(),
                },
            ]
            pax = [{"type": "adult"} for _ in range(params.passengers)]

            body = {
                "data": {
                    "slices": slices,
                    "passengers": pax,
                    "cabin_class": params.cabin.lower(),
                }
            }

            try:
                resp = requests.post(
                    f"{DUFFEL_API_URL}/offer_requests?return_offers=true",
                    headers=headers,
                    json=body,
                    timeout=30,
                )
            except Exception as e:
                print("Duffel network error for", dep, "to", ret, ":", e)
                continue

            if resp.status_code >= 400:
                print("Duffel error response", resp.status_code, resp.text)
                # We continue to the next date pair instead of failing everything
                continue

            try:
                data = resp.json()
            except Exception as e:
                print("Duffel JSON parse error:", e, "raw:", resp.text)
                continue

            offer_request = data.get("data", {})
            offers = offer_request.get("offers", [])

            for idx, offer in enumerate(offers):
                option = map_duffel_offer_to_option(offer, dep, ret, idx)
                offers_collected.append(option)

        if not offers_collected:
            return {
                "status": "ok",
                "source": "duffel",
                "options": [],
            }

        filtered = apply_filters(offers_collected, params)

        return {
            "status": "ok",
            "source": "duffel",
            "options": [o.dict() for o in filtered],
        }

    except Exception as e:
        print("Unexpected error in search_business with Duffel:", e)
        raise HTTPException(status_code=500, detail="Duffel API error")


# ------------- Admin credits endpoint ------------- #

USER_WALLETS: dict[str, int] = {}


@app.post("/admin/add-credits")
def admin_add_credits(
    payload: CreditUpdateRequest,
    x_admin_token: str = Header(None, alias="X-Admin-Token"),
):
    # Debug logging for token mismatch investigation
    print("DEBUG_received_token:", repr(x_admin_token))
    print("DEBUG_expected_token:", repr(ADMIN_API_TOKEN))

    # Normalise values
    received = (x_admin_token or "").strip()
    expected = (ADMIN_API_TOKEN or "").strip()

    if received.lower().startswith("bearer "):
        received = received[7:].strip()

    if expected == "":
        raise HTTPException(status_code=500, detail="Admin token not configured")

    if received != expected:
        raise HTTPException(status_code=401, detail="Invalid admin token")

    # Accept any field name for the credit change
    change_amount = (
        payload.delta
        if payload.delta is not None
        else payload.amount
        if payload.amount is not None
        else payload.creditAmount
        if payload.creditAmount is not None
        else payload.value
    )

    if change_amount is None:
        raise HTTPException(
            status_code=400,
            detail="Missing credit amount. Expected one of: amount, delta, creditAmount, value.",
        )

    current = USER_WALLETS.get(payload.userId, 0)
    new_balance = max(0, current + change_amount)
    USER_WALLETS[payload.userId] = new_balance

    return {
        "userId": payload.userId,
        "newBalance": new_balance,
    }


# ------------- Duffel test endpoint (uses v2 HTTP) ------------- #

@app.get("/duffel-test")
def duffel_test(
    origin: str,
    destination: str,
    departure: date,
    passengers: int = 1,
):
    """
    Simple test endpoint for Duffel search.
    Uses DUFFEL_ACCESS_TOKEN and Duffel API v2.
    No bookings are created.
    """

    if not DUFFEL_ACCESS_TOKEN:
        raise HTTPException(status_code=500, detail="Duffel not configured")

    headers = duffel_headers()

    slices = [{
        "origin": origin,
        "destination": destination,
        "departure_date": departure.isoformat(),
    }]
    pax = [{"type": "adult"} for _ in range(passengers)]

    body = {
        "data": {
            "slices": slices,
            "passengers": pax,
            "cabin_class": "business",
        }
    }

    try:
        resp = requests.post(
            f"{DUFFEL_API_URL}/offer_requests?return_offers=true",
            headers=headers,
            json=body,
            timeout=30,
        )
    except Exception as e:
        print("Duffel network error in /duffel-test:", e)
        raise HTTPException(status_code=500, detail="Duffel API error")

    if resp.status_code >= 400:
        print("Duffel error in /duffel-test:", resp.status_code, resp.text)
        raise HTTPException(status_code=500, detail="Duffel API error")

    try:
        data = resp.json()
    except Exception as e:
        print("Duffel JSON parse error in /duffel-test:", e, "raw:", resp.text)
        raise HTTPException(status_code=500, detail="Duffel API error")

    offer_request = data.get("data", {})
    offers = offer_request.get("offers", [])

    results = []
    for offer in offers:
        owner = offer.get("owner", {})
        results.append({
            "id": offer["id"],
            "airline": owner.get("name"),
            "airlineCode": owner.get("iata_code"),
            "price": float(offer["total_amount"]),
            "currency": offer["total_currency"],
        })

    return {
        "status": "ok",
        "source": "duffel",
        "offers": results,
    }
