import os
from datetime import date
from typing import List, Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from amadeus import Client, ResponseError


# ------------- Models ------------- #

class SearchParams(BaseModel):
    origin: str                 # "LON"
    destination: str            # "TLV"
    earliestDeparture: date     # "2025-12-22"
    latestDeparture: date       # "2026-01-01"
    minStayDays: int            # 21
    maxStayDays: int            # 28
    maxPrice: float             # 2000
    cabin: str = "BUSINESS"
    passengers: int = 1


class FlightOption(BaseModel):
    id: str
    airline: str
    price: float
    currency: str
    departureDate: str
    returnDate: str
    stops: int
    bookingUrl: Optional[str] = None


# ------------- FastAPI app ------------- #

app = FastAPI()

# Allow your Base44 frontend to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # you can replace "*" with your Base44 domain once it is final
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------- Amadeus client ------------- #

AMADEUS_API_KEY = os.getenv("AMADEUS_API_KEY")
AMADEUS_API_SECRET = os.getenv("AMADEUS_API_SECRET")
AMADEUS_ENV = os.getenv("AMADEUS_ENV", "test")  # "test" or "production"

hostname = "production" if AMADEUS_ENV.lower() == "production" else "test"

amadeus = None
if AMADEUS_API_KEY and AMADEUS_API_SECRET:
    amadeus = Client(
        client_id=AMADEUS_API_KEY,
        client_secret=AMADEUS_API_SECRET,
        hostname=hostname,
    )


# ------------- Helpers ------------- #

def dummy_results(params: SearchParams) -> List[FlightOption]:
    """Fallback results when Amadeus returns nothing or fails."""
    return [
        FlightOption(
            id="TK123",
            airline="Turkish Airlines",
            price=1299,
            currency="GBP",
            departureDate=params.earliestDeparture.isoformat(),
            returnDate=params.latestDeparture.isoformat(),
            stops=1,
            bookingUrl="https://www.turkishairlines.com",
        ),
        FlightOption(
            id="LH456",
            airline="Lufthansa",
            price=1550,
            currency="GBP",
            departureDate=params.earliestDeparture.isoformat(),
            returnDate=params.latestDeparture.isoformat(),
            stops=1,
            bookingUrl="https://www.lufthansa.com",
        ),
    ]


def map_amadeus_offer_to_option(offer, params: SearchParams, index: int) -> FlightOption:
    price = float(offer["price"]["grandTotal"])
    currency = offer["price"]["currency"]

    # Take first segments for basic display
    first_itinerary = offer["itineraries"][0]
    last_itinerary = offer["itineraries"][-1]

    departure_date = first_itinerary["segments"][0]["departure"]["at"][:10]
    return_date = last_itinerary["segments"][-1]["arrival"]["at"][:10]

    # Use first validating airline if available
    airline_code = ""
    if "validatingAirlineCodes" in offer and offer["validatingAirlineCodes"]:
        airline_code = offer["validatingAirlineCodes"][0]

    stops_outbound = len(first_itinerary["segments"]) - 1

    return FlightOption(
        id=f"offer_{index}",
        airline=airline_code or "Airline",
        price=price,
        currency=currency,
        departureDate=departure_date,
        returnDate=return_date,
        stops=stops_outbound,
        # For MVP, bookingUrl is just placeholder
        bookingUrl=None,
    )


# ------------- Routes ------------- #

@app.get("/")
def home():
    return {"message": "Flyvo backend is running"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/search-business")
def search_business(params: SearchParams):
    """
    Main endpoint used by the Base44 frontend.
    Tries Amadeus first, falls back to dummy data if needed.
    """

    # If Amadeus is not configured, return dummy flights
    if amadeus is None:
        return {
            "status": "ok",
            "source": "dummy",
            "options": [o.dict() for o in dummy_results(params)],
        }

    try:
        # Simple MVP logic
        # Use earliestDeparture as departureDate and latestDeparture as returnDate
        response = amadeus.shopping.flight_offers_search.get(
            originLocationCode=params.origin,
            destinationLocationCode=params.destination,
            departureDate=params.earliestDeparture.isoformat(),
            returnDate=params.latestDeparture.isoformat(),
            adults=params.passengers,
            travelClass=params.cabin,
            currencyCode="GBP",
            max=20,
        )

        offers = response.data or []

        # Map to FlightOption objects
        mapped = [
            map_amadeus_offer_to_option(offer, params, i)
            for i, offer in enumerate(offers)
        ]

        # Filter by maxPrice
        filtered = [m for m in mapped if m.price <= params.maxPrice]

        # Sort by price
        filtered.sort(key=lambda x: x.price)

        if not filtered:
            # No offer under the budget, let the frontend show "no results"
            return {
                "status": "no_results",
                "source": "amadeus",
                "options": [],
            }

        return {
            "status": "ok",
            "source": "amadeus",
            "options": [o.dict() for o in filtered],
        }

    except ResponseError as e:
        # Amadeus error, log friendly message and return dummy data
        print("Amadeus API error:", e)
        return {
            "status": "ok",
            "source": "dummy_on_error",
            "options": [o.dict() for o in dummy_results(params)],
        }
    except Exception as e:
        # Any other error, same fallback
        print("Unexpected error in search_business:", e)
        return {
            "status": "ok",
            "source": "dummy_on_error",
            "options": [o.dict() for o in dummy_results(params)],
        }
