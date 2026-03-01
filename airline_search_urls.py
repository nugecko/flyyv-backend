"""
airline_search_urls.py

Deep-link URL templates for the top 30 Business Class airlines.
Each template builds a pre-filled search URL on the airline's own website.

Cabin class codes vary by airline:
  C = Business (most airlines)
  J = Business (some)
  W = Premium Economy

Date formats vary:
  YYYY-MM-DD  — most common
  YYYYMMDD    — some (BA, etc.)
  DD/MM/YYYY  — rare

Usage:
    from airline_search_urls import build_airline_search_url
    url = build_airline_search_url("BA", "LHR", "JFK", date(2026,3,1), date(2026,3,15), "BUSINESS", 1)
"""

from datetime import date
from typing import Optional
from urllib.parse import urlencode


def _fmt(d: date, fmt: str) -> str:
    return d.strftime(fmt)


AIRLINE_HOMEPAGES = {
    # Middle East
    "EK": "https://www.emirates.com",
    "QR": "https://www.qatarairways.com",
    "EY": "https://www.etihad.com",
    "RJ": "https://www.rj.com",
    "WY": "https://www.omanair.com",
    "ME": "https://www.mea.com.lb",
    "MS": "https://www.egyptair.com",
    # Europe - Western
    "BA": "https://www.britishairways.com",
    "VS": "https://www.virginatlantic.com",
    "AF": "https://www.airfrance.com",
    "KL": "https://www.klm.com",
    "LH": "https://www.lufthansa.com",
    "LX": "https://www.swiss.com",
    "OS": "https://www.austrian.com",
    "SN": "https://www.brusselsairlines.com",
    "IB": "https://www.iberia.com",
    "TP": "https://www.flytap.com",
    "SK": "https://www.flysas.com",
    "AY": "https://www.finnair.com",
    "AZ": "https://www.ita-airways.com",
    # Europe - Eastern
    "LO": "https://www.lot.com",
    "LY": "https://www.elal.com",
    # Turkey
    "TK": "https://www.turkishairlines.com",
    # Asia-Pacific
    "SQ": "https://www.singaporeair.com",
    "CX": "https://www.cathaypacific.com",
    "KE": "https://www.koreanair.com",
    "OZ": "https://www.asiana.com",
    "NH": "https://www.ana.co.jp/en",
    "JL": "https://www.jal.com",
    # North America
    "AA": "https://www.aa.com",
    "DL": "https://www.delta.com",
    "UA": "https://www.united.com",
    "AC": "https://www.aircanada.com",
    # Africa
    "ET": "https://www.ethiopianairlines.com",
    "AT": "https://www.royalairmaroc.com",
    # North Atlantic
    "FI": "https://www.icelandair.com",
}


def build_airline_search_url(
    airline_code: str,
    origin: str,
    destination: str,
    dep_date: date,
    ret_date: date,
    cabin: str,
    passengers: int,
) -> Optional[str]:
    """
    Returns the airline's homepage URL.
    Deep links are unreliable across airlines; homepage is shown as the CTA.
    """
    return AIRLINE_HOMEPAGES.get((airline_code or "").upper())


# =====================================================================
# CABIN CLASS HELPERS
# =====================================================================

def _cabin_ba(cabin: str) -> str:
    return {"FIRST": "F", "BUSINESS": "C", "PREMIUM_ECONOMY": "W", "ECONOMY": "M"}.get(cabin, "C")

def _cabin_qr(cabin: str) -> str:
    return {"FIRST": "F", "BUSINESS": "C", "PREMIUM_ECONOMY": "W", "ECONOMY": "Y"}.get(cabin, "C")

def _cabin_ek(cabin: str) -> str:
    return {"FIRST": "F", "BUSINESS": "J", "PREMIUM_ECONOMY": "W", "ECONOMY": "Y"}.get(cabin, "J")

def _cabin_lh(cabin: str) -> str:
    return {"FIRST": "F", "BUSINESS": "B", "PREMIUM_ECONOMY": "E", "ECONOMY": "Y"}.get(cabin, "B")

def _cabin_af(cabin: str) -> str:
    return {"FIRST": "F", "BUSINESS": "C", "PREMIUM_ECONOMY": "W", "ECONOMY": "Y"}.get(cabin, "C")

def _cabin_vs(cabin: str) -> str:
    return {"FIRST": "F", "BUSINESS": "U", "PREMIUM_ECONOMY": "W", "ECONOMY": "M"}.get(cabin, "U")

def _cabin_sq(cabin: str) -> str:
    return {"FIRST": "F", "BUSINESS": "J", "PREMIUM_ECONOMY": "W", "ECONOMY": "Y"}.get(cabin, "J")

def _cabin_tk(cabin: str) -> str:
    return {"FIRST": "F", "BUSINESS": "C", "PREMIUM_ECONOMY": "W", "ECONOMY": "Y"}.get(cabin, "C")

def _cabin_name(cabin: str) -> str:
    """Human-readable cabin for airlines that use text params."""
    return {"FIRST": "first", "BUSINESS": "business", "PREMIUM_ECONOMY": "premium_economy", "ECONOMY": "economy"}.get(cabin, "business")

def _cabin_name_cap(cabin: str) -> str:
    return {"FIRST": "First", "BUSINESS": "Business", "PREMIUM_ECONOMY": "PremiumEconomy", "ECONOMY": "Economy"}.get(cabin, "Business")


# =====================================================================
# INDIVIDUAL AIRLINE URL BUILDERS
# =====================================================================

def _ba(o, d, dep, ret, cabin, pax):
    # BA schedules path bypasses session lock, allows Select and Book
    return (
        f"https://www.britishairways.com/travel/schedulesresults/public/en_gb"
        f"?origin={o}&destination={d}"
        f"&outboundDate={_fmt(dep, '%Y-%m-%d')}"
        f"&inboundDate={_fmt(ret, '%Y-%m-%d')}"
        f"&CabinCode={_cabin_ba(cabin)}&NumberOfAdults={pax}&TripType=R"
    )

def _qr(o, d, dep, ret, cabin, pax):
    return (
        f"https://www.qatarairways.com/en-gb/flights.html"
        f"?bookingClass={_cabin_qr(cabin)}&tripType=R"
        f"&from={o}&to={d}"
        f"&departing={_fmt(dep, '%Y-%m-%d')}"
        f"&returning={_fmt(ret, '%Y-%m-%d')}"
        f"&adults={pax}&adolescents=0&children=0&infants=0"
    )

def _ek(o, d, dep, ret, cabin, pax):
    # Emirates requires DDMMYY flat format (no dashes), cabin lowercase
    cabin_ek = {"FIRST": "first", "BUSINESS": "business", "PREMIUM_ECONOMY": "premiumeconomy", "ECONOMY": "economy"}.get(cabin, "business")
    return (
        f"https://www.emirates.com/english/book/flights/"
        f"?origin={o}&destination={d}"
        f"&departureDate={_fmt(dep, '%d%m%y')}"
        f"&returnDate={_fmt(ret, '%d%m%y')}"
        f"&adults={pax}&cabin={cabin_ek}"
    )

def _lh(o, d, dep, ret, cabin, pax):
    return (
        f"https://www.lufthansa.com/gb/en/flight-search"
        f"#/flightSearch?origin={o}&destination={d}"
        f"&outwardDate={_fmt(dep, '%Y-%m-%d')}"
        f"&returnDate={_fmt(ret, '%Y-%m-%d')}"
        f"&adult={pax}&cabin={_cabin_lh(cabin)}&tripType=R"
    )

def _af(o, d, dep, ret, cabin, pax):
    pax_str = f"ADT:{pax}"
    return (
        f"https://wwws.airfrance.co.uk/search/offers"
        f"?pax={pax_str}&bookingFlow=LEISURE&cabin={_cabin_af(cabin)}"
        f"&tripType=R&from={o}&to={d}"
        f"&outboundDate={_fmt(dep, '%Y-%m-%d')}"
        f"&inboundDate={_fmt(ret, '%Y-%m-%d')}"
    )

def _kl(o, d, dep, ret, cabin, pax):
    pax_str = f"ADT:{pax}"
    return (
        f"https://www.klm.com/search/offers"
        f"?pax={pax_str}&bookingFlow=LEISURE&cabin={_cabin_af(cabin)}"
        f"&tripType=R&from={o}&to={d}"
        f"&outboundDate={_fmt(dep, '%Y-%m-%d')}"
        f"&inboundDate={_fmt(ret, '%Y-%m-%d')}"
    )

def _vs(o, d, dep, ret, cabin, pax):
    return (
        f"https://www.virginatlantic.com/flights/search"
        f"?originAirportCode={o}&destinationAirportCode={d}"
        f"&departureDate={_fmt(dep, '%Y-%m-%d')}"
        f"&returnDate={_fmt(ret, '%Y-%m-%d')}"
        f"&cabin={_cabin_vs(cabin)}&adults={pax}&tripType=R"
    )

def _sq(o, d, dep, ret, cabin, pax):
    return (
        f"https://www.singaporeair.com/en_UK/gb/plan-travel/our-flights/flight-search/"
        f"?tripType=R&origin={o}&destination={d}"
        f"&departureDate={_fmt(dep, '%Y-%m-%d')}"
        f"&returnDate={_fmt(ret, '%Y-%m-%d')}"
        f"&cabinClass={_cabin_name_cap(cabin)}&noofadults={pax}"
    )

def _tk(o, d, dep, ret, cabin, pax):
    # Turkish uses their booking search form with specific params
    return (
        f"https://www.turkishairlines.com/en-int/booking/flight-search/"
        f"?tripType=R&origin={o}&destination={d}"
        f"&departureDate={_fmt(dep, '%Y-%m-%d')}"
        f"&returnDate={_fmt(ret, '%Y-%m-%d')}"
        f"&adult={pax}&cabin={_cabin_tk(cabin)}"
    )

def _fi(o, d, dep, ret, cabin, pax):
    # Icelandair
    cabin_fi = {"FIRST": "business", "BUSINESS": "business", "PREMIUM_ECONOMY": "saga", "ECONOMY": "economy"}.get(cabin, "business")
    return (
        f"https://www.icelandair.com/flights/roundtrip/{o}/{d}/"
        f"{_fmt(dep, '%Y-%m-%d')}/{_fmt(ret, '%Y-%m-%d')}"
        f"/{pax}/0/0/{cabin_fi}/"
    )

def _at(o, d, dep, ret, cabin, pax):
    # Royal Air Maroc
    return (
        f"https://www.royalairmaroc.com/gb-en/book-your-flight"
        f"?type=RT&from={o}&to={d}"
        f"&departure={_fmt(dep, '%Y-%m-%d')}"
        f"&return={_fmt(ret, '%Y-%m-%d')}"
        f"&adults={pax}&cabin={_cabin_name(cabin)}"
    )

def _ey(o, d, dep, ret, cabin, pax):
    return (
        f"https://www.etihad.com/en-gb/book/flights/"
        f"?tripType=R&from={o}&to={d}"
        f"&departDate={_fmt(dep, '%Y-%m-%d')}"
        f"&returnDate={_fmt(ret, '%Y-%m-%d')}"
        f"&adults={pax}&class={_cabin_name(cabin)}"
    )

def _ib(o, d, dep, ret, cabin, pax):
    return (
        f"https://www.iberia.com/flights/?round-trip"
        f"&origin1={o}&destination1={d}"
        f"&departure1={_fmt(dep, '%Y-%m-%d')}"
        f"&return1={_fmt(ret, '%Y-%m-%d')}"
        f"&adults={pax}&class={_cabin_qr(cabin)}"
    )

def _ay(o, d, dep, ret, cabin, pax):
    return (
        f"https://www.finnair.com/gb-en/flight-search/results"
        f"?type=RETURN&from={o}&to={d}"
        f"&departureDate={_fmt(dep, '%Y-%m-%d')}"
        f"&returnDate={_fmt(ret, '%Y-%m-%d')}"
        f"&adults={pax}&cabinClass={_cabin_name(cabin)}"
    )

def _lo(o, d, dep, ret, cabin, pax):
    return (
        f"https://www.lot.com/gb/en/flight-search"
        f"#?tripType=RT&origin={o}&destination={d}"
        f"&departureDate={_fmt(dep, '%Y-%m-%d')}"
        f"&returnDate={_fmt(ret, '%Y-%m-%d')}"
        f"&adults={pax}&cabin={_cabin_ba(cabin)}"
    )

def _tp(o, d, dep, ret, cabin, pax):
    return (
        f"https://www.flytap.com/en-gb/flight-search"
        f"?tripType=RT&from={o}&to={d}"
        f"&departure={_fmt(dep, '%Y-%m-%d')}"
        f"&return={_fmt(ret, '%Y-%m-%d')}"
        f"&adults={pax}&cabin={_cabin_name(cabin)}"
    )

def _ly(o, d, dep, ret, cabin, pax):
    return (
        f"https://www.elal.com/en/Flights/Search-Results/"
        f"?type=RT&origin={o}&destination={d}"
        f"&departure={_fmt(dep, '%Y-%m-%d')}"
        f"&return={_fmt(ret, '%Y-%m-%d')}"
        f"&adults={pax}&cabin={_cabin_name(cabin)}"
    )

def _sk(o, d, dep, ret, cabin, pax):
    return (
        f"https://www.flysas.com/en/search/"
        f"?type=roundtrip&origin={o}&destination={d}"
        f"&outbound={_fmt(dep, '%Y-%m-%d')}"
        f"&inbound={_fmt(ret, '%Y-%m-%d')}"
        f"&adults={pax}&cabin={_cabin_name(cabin)}"
    )

def _lx(o, d, dep, ret, cabin, pax):
    return (
        f"https://www.swiss.com/gb/en/book/flights"
        f"#/?origin={o}&destination={d}"
        f"&outwardDate={_fmt(dep, '%Y-%m-%d')}"
        f"&returnDate={_fmt(ret, '%Y-%m-%d')}"
        f"&adults={pax}&cabin={_cabin_lh(cabin)}&tripType=R"
    )

def _os(o, d, dep, ret, cabin, pax):
    return (
        f"https://www.austrian.com/gb/en/book/flights"
        f"#/?origin={o}&destination={d}"
        f"&outwardDate={_fmt(dep, '%Y-%m-%d')}"
        f"&returnDate={_fmt(ret, '%Y-%m-%d')}"
        f"&adults={pax}&cabin={_cabin_lh(cabin)}&tripType=R"
    )

def _sn(o, d, dep, ret, cabin, pax):
    return (
        f"https://www.brusselsairlines.com/gb/en/book/flights"
        f"#/?origin={o}&destination={d}"
        f"&outwardDate={_fmt(dep, '%Y-%m-%d')}"
        f"&returnDate={_fmt(ret, '%Y-%m-%d')}"
        f"&adults={pax}&cabin={_cabin_lh(cabin)}&tripType=R"
    )

def _cy(o, d, dep, ret, cabin, pax):
    # Cyprus Airways — fallback search page
    return (
        f"https://www.cyprusairways.com/flights"
        f"?origin={o}&destination={d}"
        f"&departure={_fmt(dep, '%Y-%m-%d')}"
        f"&return={_fmt(ret, '%Y-%m-%d')}"
        f"&adults={pax}&cabin={_cabin_name(cabin)}"
    )

def _cx(o, d, dep, ret, cabin, pax):
    return (
        f"https://www.cathaypacific.com/cx/en_GB/booking/flight-search.html"
        f"?origin={o}&destination={d}"
        f"&departureDate={_fmt(dep, '%Y-%m-%d')}"
        f"&returnDate={_fmt(ret, '%Y-%m-%d')}"
        f"&cabin={_cabin_sq(cabin)}&adults={pax}&tripType=R"
    )

def _aa(o, d, dep, ret, cabin, pax):
    return (
        f"https://www.aa.com/booking/search"
        f"?cabin={_cabin_name(cabin)}&trips[0][orig]={o}&trips[0][dest]={d}"
        f"&trips[0][date]={_fmt(dep, '%Y-%m-%d')}"
        f"&trips[1][orig]={d}&trips[1][dest]={o}"
        f"&trips[1][date]={_fmt(ret, '%Y-%m-%d')}"
        f"&pax[adt]={pax}&type=RT"
    )

def _dl(o, d, dep, ret, cabin, pax):
    cabin_dl = {"FIRST": "F", "BUSINESS": "C", "PREMIUM_ECONOMY": "W", "ECONOMY": "Y"}.get(cabin, "C")
    return (
        f"https://www.delta.com/flightsearch/book-a-flight"
        f"?action=findFlights&originCity={o}&destinationCity={d}"
        f"&departureDate={_fmt(dep, '%Y-%m-%d')}"
        f"&returnDate={_fmt(ret, '%Y-%m-%d')}"
        f"&paxCount={pax}&cabinClass={cabin_dl}"
    )

def _ua(o, d, dep, ret, cabin, pax):
    # United FSR path - requires tt=2 (roundtrip), at=2 (adult), st=bestf
    cabin_ua = {"FIRST": "F", "BUSINESS": "C", "PREMIUM_ECONOMY": "W", "ECONOMY": "Y"}.get(cabin, "C")
    return (
        f"https://www.united.com/en/us/fsr/chooseFlights"
        f"?f={o}&t={d}"
        f"&d={_fmt(dep, '%Y-%m-%d')}"
        f"&r={_fmt(ret, '%Y-%m-%d')}"
        f"&tt=2&at=2&st=bestf&c={cabin_ua}"
    )

def _ac(o, d, dep, ret, cabin, pax):
    cabin_ac = {"FIRST": "business", "BUSINESS": "business", "PREMIUM_ECONOMY": "premium", "ECONOMY": "economy"}.get(cabin, "business")
    return (
        f"https://www.aircanada.com/gb/en/aco/home/book/flights-only/results.html"
        f"#/search?org0={o}&dest0={d}&outDate0={_fmt(dep, '%Y-%m-%d')}"
        f"&inDate0={_fmt(ret, '%Y-%m-%d')}&ADT={pax}&YTH=0&CHD=0&INF=0"
        f"&cabin={cabin_ac}&tripType=O"
    )

def _rj(o, d, dep, ret, cabin, pax):
    return (
        f"https://www.rj.com/en/book-a-flight"
        f"?origin={o}&destination={d}"
        f"&departure={_fmt(dep, '%Y-%m-%d')}"
        f"&return={_fmt(ret, '%Y-%m-%d')}"
        f"&adults={pax}&cabin={_cabin_name(cabin)}&tripType=R"
    )

def _wy(o, d, dep, ret, cabin, pax):
    return (
        f"https://www.omanair.com/en/book-flight"
        f"?origin={o}&destination={d}"
        f"&departure={_fmt(dep, '%Y-%m-%d')}"
        f"&return={_fmt(ret, '%Y-%m-%d')}"
        f"&adults={pax}&cabin={_cabin_name(cabin)}&tripType=R"
    )

def _me(o, d, dep, ret, cabin, pax):
    return (
        f"https://flights.mea.com.lb/en"
        f"?from={o}&to={d}"
        f"&departure={_fmt(dep, '%Y-%m-%d')}"
        f"&return={_fmt(ret, '%Y-%m-%d')}"
        f"&adults={pax}&cabin={_cabin_name(cabin)}&type=RT"
    )

def _ms(o, d, dep, ret, cabin, pax):
    # EgyptAir
    return (
        f"https://www.egyptair.com/en/fly/Pages/Book-Now.aspx"
        f"?origin={o}&destination={d}"
        f"&departure={_fmt(dep, '%Y-%m-%d')}"
        f"&return={_fmt(ret, '%Y-%m-%d')}"
        f"&adults={pax}&cabin={_cabin_name(cabin)}&tripType=R"
    )

def _et(o, d, dep, ret, cabin, pax):
    # Ethiopian Airlines
    return (
        f"https://www.ethiopianairlines.com/aa/book/flight-selection"
        f"?from={o}&to={d}"
        f"&departure={_fmt(dep, '%Y-%m-%d')}"
        f"&return={_fmt(ret, '%Y-%m-%d')}"
        f"&adults={pax}&cabin={_cabin_name(cabin)}&type=roundtrip"
    )


# =====================================================================
# REGISTRY
# =====================================================================

AIRLINE_URL_BUILDERS = {
    # Middle East
    "EK": _ek,   # Emirates
    "QR": _qr,   # Qatar Airways
    "EY": _ey,   # Etihad
    "RJ": _rj,   # Royal Jordanian
    "WY": _wy,   # Oman Air
    "ME": _me,   # Middle East Airlines
    "MS": _ms,   # EgyptAir

    # Europe - Western
    "BA": _ba,   # British Airways
    "VS": _vs,   # Virgin Atlantic
    "AF": _af,   # Air France
    "KL": _kl,   # KLM
    "LH": _lh,   # Lufthansa
    "LX": _lx,   # Swiss
    "OS": _os,   # Austrian
    "SN": _sn,   # Brussels Airlines
    "IB": _ib,   # Iberia
    "TP": _tp,   # TAP Portugal
    "SK": _sk,   # SAS
    "AY": _ay,   # Finnair

    # Europe - Eastern
    "LO": _lo,   # LOT Polish Airlines
    "LY": _ly,   # El Al

    # Turkey
    "TK": _tk,   # Turkish Airlines

    # Asia-Pacific
    "SQ": _sq,   # Singapore Airlines
    "CX": _cx,   # Cathay Pacific

    # North America
    "AA": _aa,   # American Airlines
    "DL": _dl,   # Delta Air Lines
    "UA": _ua,   # United Airlines
    "AC": _ac,   # Air Canada

    # Africa & North Africa
    "ET": _et,   # Ethiopian Airlines
    "AT": _at,   # Royal Air Maroc

    # North Atlantic
    "FI": _fi,   # Icelandair
}


# =====================================================================
# HELPER: get all booking options for a result
# =====================================================================

def _build_google_flights_url(
    origin: str,
    destination: str,
    dep_date: date,
    ret_date: date,
    cabin_class: int,
    adults: int,
) -> str:
    """
    Build a Google Flights URL with a properly encoded tfs protobuf parameter.
    Reverse-engineered from real Google Flights URLs — exactly matches their format.

    cabin_class: 1=economy, 2=premium economy, 3=business, 4=first
    Works with 3-letter IATA airport codes only (not city/metro codes).
    """
    import base64

    def _iata_bytes(code: str) -> bytes:
        return bytes([0x08, 0x01, 0x12, 0x03]) + code.upper().encode()

    def _leg(date_str: str, orig: str, dest: str) -> bytes:
        date_b = date_str.encode()       # e.g. b'2026-03-16', 10 bytes
        orig_b = _iata_bytes(orig)       # 7 bytes
        dest_b = _iata_bytes(dest)       # 7 bytes
        inner = (
            bytes([0x12, len(date_b)]) + date_b +
            bytes([0x6a, len(orig_b)]) + orig_b +
            bytes([0x72, len(dest_b)]) + dest_b
        )
        return bytes([0x1a, len(inner)]) + inner

    dep_str = dep_date.strftime("%Y-%m-%d")
    ret_str = ret_date.strftime("%Y-%m-%d")

    payload = (
        bytes([0x08, 0x1c, 0x10, cabin_class]) +
        _leg(dep_str, origin, destination) +
        _leg(ret_str, destination, origin) +
        bytes([0x42, 0x01, 0x01, 0x48, adults, 0x70, 0x01, 0x98, 0x01, 0x01, 0xc8, 0x01, 0x01])
    )

    tfs = base64.urlsafe_b64encode(payload).rstrip(b"=").decode()
    return f"https://www.google.com/travel/flights?tfs={tfs}&hl=en&curr=GBP"


def get_booking_urls(
    airline_code: str,
    origin: str,
    destination: str,
    dep_date: date,
    ret_date: date,
    cabin: str,
    passengers: int,
    price: float,
    currency: str,
) -> dict:
    """
    Returns a dict with all booking URLs for a flight result:
    {
        "skyscanner": "https://...",
        "kayak": "https://...",
        "airline": "https://..." or None,
        "price": 1234.56,
        "currency": "GBP"
    }
    """
    dep_str = dep_date.strftime("%Y%m%d")
    ret_str = ret_date.strftime("%Y%m%d")

    # Normalise cabin to uppercase with underscores
    cabin_norm = (cabin or "BUSINESS").strip().upper().replace(" ", "_").replace("-", "_")
    # Also handle lowercase inputs like "business", "economy"
    cabin_norm_map = {
        "BUSINESS": "BUSINESS", "FIRST": "FIRST",
        "PREMIUM_ECONOMY": "PREMIUM_ECONOMY", "ECONOMY": "ECONOMY",
        # lowercase variants
        "business": "BUSINESS", "first": "FIRST",
        "premium_economy": "PREMIUM_ECONOMY", "economy": "ECONOMY",
        "premiumeconomy": "PREMIUM_ECONOMY",
    }
    cabin_norm = cabin_norm_map.get(cabin_norm, "BUSINESS")

    # Skyscanner cabin values
    cabin_sky = {"BUSINESS": "business", "FIRST": "first", "PREMIUM_ECONOMY": "premiumeconomy", "ECONOMY": "economy"}.get(cabin_norm, "business")
    # Kayak cabin values — premium economy is "premium" not "premiumeconomy"
    cabin_kayak = {"BUSINESS": "business", "FIRST": "first", "PREMIUM_ECONOMY": "premium", "ECONOMY": "economy"}.get(cabin_norm, "business")
    # Google Flights class: 1=economy, 2=premium economy, 3=business, 4=first
    cabin_google_int = {"ECONOMY": 1, "PREMIUM_ECONOMY": 2, "BUSINESS": 3, "FIRST": 4}.get(cabin_norm, 3)

    pax = max(1, int(passengers or 1))

    # Google Flights — build proper tfs protobuf parameter.
    # Reverse-engineered from real Google Flights URLs. Exactly matches Google's own format.
    google = _build_google_flights_url(origin, destination, dep_date, ret_date, cabin_google_int, pax)

    # Skyscanner — correct param is adultsv2, not adults
    skyscanner = (
        f"https://www.skyscanner.net/transport/flights"
        f"/{origin}/{destination}/{dep_str}/{ret_str}"
        f"/?adultsv2={pax}&cabinclass={cabin_sky}&childrenv2=&ref=flyyv&rtn=1"
    )

    kayak = (
        f"https://www.kayak.co.uk/flights"
        f"/{origin}-{destination}/{dep_date.strftime('%Y-%m-%d')}/{ret_date.strftime('%Y-%m-%d')}"
        f"/{cabin_kayak}/{pax}adults"
    )

    airline_url = build_airline_search_url(
        airline_code, origin, destination, dep_date, ret_date, cabin_norm, pax
    )

    return {
        "skyscanner": skyscanner,
        "kayak": kayak,
        "google": google,
        "airline": airline_url,
        "price": price,
        "currency": currency,
    }