"""
providers/factory.py

Routes search calls to the correct provider based on FLIGHT_PROVIDER env var.

Currently supported values:
  ttn      — tickets.ua (default, production)
  flightapi — Flight API (Skyscanner-style, stub, not yet live)

To switch providers without code changes:
  dokku config:set flyyv FLIGHT_PROVIDER=flightapi
  dokku config:set flyyv FLIGHTAPI_KEY=your_key

Revert instantly:
  dokku config:set flyyv FLIGHT_PROVIDER=ttn
"""

from datetime import date
from typing import List, Optional

from config import FLIGHT_PROVIDER
from schemas.search import FlightOption, SearchParams


def run_provider_scan(
    params: SearchParams,
    dep_override: Optional[date] = None,
    ret_override: Optional[date] = None,
) -> List[FlightOption]:
    """
    Canonical entry point for all search and alert scanning.

    Replaces all direct calls to run_ttn_scan() throughout the codebase.
    All callers (process_date_pair_offers, process_alert, etc.) should call this.
    """
    provider = FLIGHT_PROVIDER

    if provider == "flightapi":
        from providers.flightapi import run_flightapi_scan
        return run_flightapi_scan(params, dep_override=dep_override, ret_override=ret_override)

    # Default: TTN
    from providers.ttn import run_ttn_scan
    return run_ttn_scan(params, dep_override=dep_override, ret_override=ret_override)
