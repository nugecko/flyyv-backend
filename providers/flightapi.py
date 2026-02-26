"""
providers/flightapi.py

Flight API (Skyscanner-style) adapter.

Status: STUB — not yet implemented.

To activate:
  1. Set FLIGHT_PROVIDER=flightapi in Dokku env.
  2. Set FLIGHTAPI_KEY=your_key_here.
  3. Implement search_roundtrip_flightapi() below.

Key differences from TTN:
  - Booking URL is returned inline in search results (no separate booking step).
  - Users click a direct handoff URL — no passport collection required.
  - Pricing is typically per passenger already.
  - Cabin mapping: "Economy", "Business", "First", "Premium_Economy"
  - Error codes: 200 OK, 404 not found, 410 no service, 429 rate limit

Flight API endpoint (roundtrip):
  POST https://api.flightapi.io/roundtrip
  Body: { key, origin, destination, departure, return, adults, children, infants, type, currency, region }
  Cost: 2 credits per call
"""

from datetime import date
from typing import List, Optional

from schemas.search import FlightOption, SearchParams


def run_flightapi_scan(
    params: SearchParams,
    dep_override: Optional[date] = None,
    ret_override: Optional[date] = None,
) -> List[FlightOption]:
    """
    Entry point called by providers/factory.py when FLIGHT_PROVIDER=flightapi.
    Must return the same List[FlightOption] shape as run_ttn_scan.
    """
    raise NotImplementedError(
        "Flight API provider is not yet implemented. "
        "Set FLIGHT_PROVIDER=ttn to use TTN, or implement this function."
    )


def search_roundtrip_flightapi(
    origin: str,
    destination: str,
    departure: date,
    return_date: date,
    passengers: int,
    cabin: str,
    currency: str = "GBP",
) -> List[FlightOption]:
    """
    Single-pair roundtrip search via Flight API.
    Returns normalised FlightOption list, or empty list on failure.

    TODO:
    1. POST to https://api.flightapi.io/roundtrip
    2. Walk itineraries → legs → segments → places/carriers/agents reference arrays
    3. Extract bookingUrl from itineraries[].pricing_options[].items[].url
    4. Map to FlightOption (price is per passenger already)
    """
    raise NotImplementedError("Not yet implemented — see docstring above.")
