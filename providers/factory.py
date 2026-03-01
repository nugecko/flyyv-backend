"""
providers/factory.py

Routes search calls to the correct provider based on FLIGHT_PROVIDER env var.

Currently supported values:
  ttn      — tickets.ua (default, production)
  flightapi — Flight API (Skyscanner-style, stub, not yet live)
  duffel   — Duffel API (current production provider)

To switch providers without code changes:
  dokku config:set flyyv FLIGHT_PROVIDER=duffel

Revert instantly:
  dokku config:set flyyv FLIGHT_PROVIDER=ttn
"""

from datetime import date
from typing import List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from config import FLIGHT_PROVIDER
from schemas.search import FlightOption, SearchParams


# Max airports to search per city code per direction.
# LON has 5 airports but for Business class only LHR+LGW matter.
# Keeping this low avoids too many parallel Duffel calls.
_MAX_AIRPORTS_PER_CITY = 2


def _run_duffel_expanded(
    params: SearchParams,
    dep_override: Optional[date],
    ret_override: Optional[date],
) -> List[FlightOption]:
    """
    Expand city codes (LON, NYC, PAR, etc.) to real airport codes,
    run a Duffel scan for each origin x destination pair in parallel,
    and merge the results.

    e.g. LON->NYC becomes LHR->JFK, LHR->EWR, LGW->JFK, LGW->EWR (4 scans).
    Without this, Duffel returns ~2 results for city codes.
    """
    from providers.duffel import run_duffel_scan
    from city_airports import get_airports_for_code

    origin_code = (getattr(params, "origin", None) or "").upper().strip()
    dest_code = (getattr(params, "destination", None) or "").upper().strip()

    origins = get_airports_for_code(origin_code)[:_MAX_AIRPORTS_PER_CITY]
    destinations = get_airports_for_code(dest_code)[:_MAX_AIRPORTS_PER_CITY]

    # If no expansion needed, run directly
    if origins == [origin_code] and destinations == [dest_code]:
        return run_duffel_scan(params, dep_override=dep_override, ret_override=ret_override)

    print(
        f"[factory] expanding city codes: {origin_code}->{origins}, "
        f"{dest_code}->{destinations}"
    )

    airport_pairs = [(o, d) for o in origins for d in destinations]
    all_results: List[FlightOption] = []
    seen_ids: set = set()

    def _scan_pair(orig: str, dest: str) -> List[FlightOption]:
        pair_params = params.model_copy(update={"origin": orig, "destination": dest})
        return run_duffel_scan(pair_params, dep_override=dep_override, ret_override=ret_override)

    with ThreadPoolExecutor(max_workers=len(airport_pairs)) as executor:
        futures = {executor.submit(_scan_pair, o, d): (o, d) for o, d in airport_pairs}
        for future in as_completed(futures):
            o, d = futures[future]
            try:
                opts = future.result() or []
                print(f"[factory] {o}->{d}: {len(opts)} offers")
                for opt in opts:
                    # Deduplicate by Duffel offer ID
                    offer_id = getattr(opt, "id", None) or id(opt)
                    if offer_id not in seen_ids:
                        seen_ids.add(offer_id)
                        all_results.append(opt)
            except Exception as e:
                print(f"[factory] scan {o}->{d} failed: {e}")

    print(f"[factory] expanded scan total merged={len(all_results)}")
    return all_results


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

    if provider == "duffel":
        return _run_duffel_expanded(params, dep_override=dep_override, ret_override=ret_override)

    # Default: TTN
    from providers.ttn import run_ttn_scan
    return run_ttn_scan(params, dep_override=dep_override, ret_override=ret_override)