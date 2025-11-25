import os
from datetime import date, datetime, timedelta
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from amadeus import Client, ResponseError
from duffel_api import Duffel


# ------------- Environment and clients ------------- #

AMADEUS_API_KEY = os.environ.get("AMADEUS_API_KEY")
AMADEUS_API_SECRET = os.environ.get("AMADEUS_API_SECRET")
AMADEUS_ENV = os.environ.get("AMADEUS_ENV", "test")

if not AMADEUS_API_KEY or not AMADEUS_API_SECRET:
    print("Warning: Amadeus credentials are not fully set")

amadeus = Client(
    client_id=AMADEUS_API_KEY,
    client_secret=AMADEUS_API_SECRET,
    hostname="test" if AMADEUS_ENV == "test" else "production",
)

DUFFEL_ACCESS_TOKEN = os.environ.get("DUFFEL_ACCESS_TOKEN")

if not DUFFEL_ACCESS_TOKEN:
    print("Warning: DUFFEL_ACCESS_TOKEN is not set")

duffel = Duffel(access_token=DUFFEL_ACCESS_TOKEN)

ADMIN_API_TOKEN = os.environ.get("ADMIN_API_TOKEN")


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
    # "flexible" = window scanning, "fixed" = one off search
    searchMode: Optional[str] = "flexible"
    # For later, for example ["duffel", "amadeus"]
    providers: Optional[List[str]] = None


class FlightOption(BaseModel):
    id: str                  # internal id
    provider: str            # "amadeus" or "duffel"
    providerOfferId: str     # id from provider
    airline: str
    airlineCode: Optional[str] = None
    price: float
    currency: str
    departure: datetime
    arrival: datetime
    stops: int
    bookingUrl: Optional[str] = None


class CreditUpdateRequest(BaseModel):
    userId: str
    amount: int  # positive or negative


class CreditBalanceResponse(BaseModel):
    userId: str
    newBalance: int


# ------------- Simple in memory credit store (Option A) ------------- #

user_balances: Dict[str, int] = {}


def get_user_balance(user_id: str) -> int:
    return user_balances.get(user_id, 0)


def update_user_balance(user_id: str, delta: int) -> int:
    new_balance = get_user_balance(user_id) + delta
    if new_balance < 0:
        new_balance = 0
    user_balances[user_id] = new_balance
    return new_balance


# ------------- FastAPI app ------------- #

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # you can tighten this later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ------------- Helper functions ------------- #

def daterange(start: date, end: date):
    current = start
    while current <= end:
        yield current
        current += timedelta(days=1)


def search_amadeus_one_way(
    origin: str,
    destination: str,
    departure_date: date,
    cabin: str,
    passengers: int,
) -> List[FlightOption]:
    try:
        response = amadeus.shopping.flight_offers_search.get(
            originLocationCode=origin,
            destinationLocationCode=destination,
            departureDate=departure_date.isoformat(),
            adults=passengers,
            travelClass=cabin,
            currencyCode="GBP",
            max=50,
        )
    except ResponseError as error:
        print(f"Amadeus error: {error}")
        return []

    offers: List[FlightOption] = []
    data = response.data

    for idx, offer in enumerate(data):
        price = float(offer["price"]["grandTotal"])
        currency = offer["price"]["currency"]

        itinerary = offer["itineraries"][0]
        segments = itinerary["segments"]
        stops = len(segments) - 1

        first_segment = segments[0]
        last_segment = segments[-1]

        departure_dt = datetime.fromisoformat(first_segment["departure"]["at"])
        arrival_dt = datetime.fromisoformat(last_segment["arrival"]["at"])

        carrier_code = first_segment["carrierCode"]
        airline_name = carrier_code

        offers.append(
            FlightOption(
                id=f"amadeus_{idx}",
                provider="amadeus",
                providerOfferId=offer["id"],
                airline=airline_name,
                airlineCode=carrier_code,
                price=price,
                currency=currency,
                departure=departure_dt,
                arrival=arrival_dt,
                stops=stops,
                bookingUrl=None,
            )
        )

    return offers


def search_duffel_one_way(
    origin: str,
    destination: str,
    departure_date: date,
    cabin: str,
    passengers: int,
) -> List[FlightOption]:
    if not DUFFEL_ACCESS_TOKEN:
        return []

    slices = [{
        "origin": origin,
        "destination": destination,
        "departure_date": departure_date.isoformat(),
    }]
    pax = [{"type": "adult"} for _ in range(passengers)]

    try:
        offer_request = duffel.offer_requests.create(
            slices=slices,
            cabin_class=cabin.lower(),
            passengers=pax,
        )
        offers_iter = duffel.offers.list(offer_request_id=offer_request.id)
    except Exception as e:
        print(f"Duffel error: {e}")
        return []

    offers: List[FlightOption] = []

    for idx, offer in enumerate(offers_iter):
        price = float(offer.total_amount)
        currency = offer.total_currency

        first_slice = offer.slices[0]
        segments = first_slice.segments
        stops = len(segments) - 1

        first_segment = segments[0]
        last_segment = segments[-1]

        departure_dt = datetime.fromisoformat(first_segment.departing_at)
        arrival_dt = datetime.fromisoformat(last_segment.arriving_at)

        owner = offer.owner

        offers.append(
            FlightOption(
                id=f"duffel_{idx}",
                provider="duffel",
                providerOfferId=offer.id,
                airline=owner.name,
                airlineCode=owner.iata_code,
                price=price,
                currency=currency,
                departure=departure_dt,
                arrival=arrival_dt,
                stops=stops,
                bookingUrl=None,
            )
        )

    return offers


def apply_price_filter(
    offers: List[FlightOption], max_price: Optional[float]
) -> List[FlightOption]:
    if max_price is None:
        return offers
    return [o for o in offers if o.price <= max_price]


# ------------- Routes ------------- #

@app.get("/")
def root():
    return {"status": "ok", "message": "Flyvo backend running"}


@app.post("/search", response_model=List[FlightOption])
def search_flights(params: SearchParams):
    """
    Simple search:
    if searchMode == "fixed" it will only search on earliestDeparture date
    if searchMode == "flexible" it will search the whole window between earliest and latest
    for each departure date it searches one way and returns all combined results
    """

    providers = params.providers or ["amadeus", "duffel"]

    all_offers: List[FlightOption] = []

    if params.searchMode == "fixed":
        dates_to_check = [params.earliestDeparture]
    else:
        dates_to_check = list(daterange(params.earliestDeparture, params.latestDeparture))

    for dep_date in dates_to_check:
        if "amadeus" in providers:
            all_offers.extend(
                search_amadeus_one_way(
                    origin=params.origin,
                    destination=params.destination,
                    departure_date=dep_date,
                    cabin=params.cabin,
                    passengers=params.passengers,
                )
            )
        if "duffel" in providers:
            all_offers.extend(
                search_duffel_one_way(
                    origin=params.origin,
                    destination=params.destination,
                    departure_date=dep_date,
                    cabin=params.cabin,
                    passengers=params.passengers,
                )
            )

    all_offers = apply_price_filter(all_offers, params.maxPrice)
    all_offers.sort(key=lambda o: o.price)

    return all_offers


@app.get("/duffel-test")
def duffel_test(
    origin: str,
    destination: str,
    departure: date,
    passengers: int = 1,
):
    """
    Simple testing endpoint for Duffel only.
    """

    offers = search_duffel_one_way(
        origin=origin,
        destination=destination,
        departure_date=departure,
        cabin="business",
        passengers=passengers,
    )
    return {"offers": offers}


@app.post("/admin/update-credits", response_model=CreditBalanceResponse)
def admin_update_credits(
    payload: CreditUpdateRequest,
    x_admin_token: str = Header(..., alias="X-Admin-Token"),
):
    """
    Admin endpoint used by Base44 tool.
    Updates the balance for a given user by the requested amount.
    """

    if not ADMIN_API_TOKEN or x_admin_token != ADMIN_API_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid admin token")

    new_balance = update_user_balance(payload.userId, payload.amount)
    return CreditBalanceResponse(userId=payload.userId, newBalance=new_balance)


@app.get("/balance/{user_id}", response_model=CreditBalanceResponse)
def get_balance(user_id: str):
    """
    Returns current balance for a user.
    """

    balance = get_user_balance(user_id)
    return CreditBalanceResponse(userId=user_id, newBalance=balance)
