import os
import smtplib
from email.message import EmailMessage
from datetime import datetime, date
from typing import List, Dict, Tuple, Any, Optional
from urllib.parse import urlencode, quote_plus

from fastapi import HTTPException

# =====================================================================
# SECTION START: SMTP CONFIG AND CONSTANTS
# =====================================================================

SMTP_HOST = os.environ.get("SMTP_HOST", "smtp.smtp2go.com")
SMTP_PORT = int(os.environ.get("SMTP_PORT", "587"))
SMTP_USERNAME = os.environ.get("SMTP_USERNAME")
SMTP_PASSWORD = os.environ.get("SMTP_PASSWORD")
ALERT_FROM_EMAIL = os.environ.get("ALERT_FROM_EMAIL", "alert@flyyv.com")

FRONTEND_BASE_URL = os.environ.get("FRONTEND_BASE_URL", "https://flyyv.com")
API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.flyyv.com")

# =====================================================================
# SECTION END: SMTP CONFIG AND CONSTANTS
# =====================================================================


# =====================================================================
# SECTION START: BOOKING URL + TRACKING HELPERS
# =====================================================================

def _build_tracked_url(
    destination_url: str,
    src: str,
    alert_id: Optional[str] = None,
    run_id: Optional[str] = None,
    airline_code: Optional[str] = None,
    origin: Optional[str] = None,
    destination: Optional[str] = None,
    dep_date: Optional[str] = None,
    ret_date: Optional[str] = None,
    cabin: Optional[str] = None,
    passengers: Optional[int] = None,
    price: Optional[float] = None,
) -> str:
    """
    Wraps any URL through api.flyyv.com/go for email click tracking.
    The backend logs the click then 302 redirects to destination_url.
    """
    base = API_BASE_URL.rstrip("/")
    params: Dict[str, str] = {"url": destination_url, "src": src}
    if alert_id:
        params["alert_id"] = str(alert_id)
    if run_id:
        params["run_id"] = str(run_id)
    if airline_code:
        params["airline"] = airline_code
    if origin:
        params["origin"] = origin
    if destination:
        params["destination"] = destination
    if dep_date:
        params["dep"] = dep_date
    if ret_date:
        params["ret"] = ret_date
    if cabin:
        params["cabin"] = cabin
    if passengers:
        params["pax"] = str(passengers)
    if price:
        params["price"] = str(int(price))
    return f"{base}/go?{urlencode(params)}"


def _get_booking_urls_for_result(
    result,
    cabin: str,
    passengers: int,
    alert_id: Optional[str] = None,
    run_id: Optional[str] = None,
) -> Dict[str, Optional[str]]:
    """
    Build tracked Skyscanner, Kayak, and airline direct URLs for an alert result.
    Returns dict with keys: skyscanner, kayak, airline (may be None).
    All URLs go through /go tracking redirect.
    """
    try:
        from airline_search_urls import get_booking_urls
        from datetime import date as date_type

        dep_raw = getattr(result, "departureDate", None)
        ret_raw = getattr(result, "returnDate", None)
        airline_code = getattr(result, "airlineCode", None) or ""
        origin = getattr(result, "origin", None) or ""
        destination = getattr(result, "destination", None) or ""

        # Parse dates
        def _to_date(v):
            if isinstance(v, date_type):
                return v
            if v:
                try:
                    return date_type.fromisoformat(str(v)[:10])
                except Exception:
                    pass
            return None

        dep_date = _to_date(dep_raw)
        ret_date = _to_date(ret_raw)

        if not (dep_date and ret_date and origin and destination):
            return {"skyscanner": None, "kayak": None, "airline": None}

        raw = get_booking_urls(
            airline_code=airline_code,
            origin=origin,
            destination=destination,
            dep_date=dep_date,
            ret_date=ret_date,
            cabin=cabin,
            passengers=passengers,
            price=getattr(result, "price", 0) or 0,
            currency="GBP",
        )

        dep_str = dep_date.isoformat()
        ret_str = ret_date.isoformat()
        price_val = getattr(result, "price", None)

        def _track(url: Optional[str], src: str) -> Optional[str]:
            if not url:
                return None
            return _build_tracked_url(
                destination_url=url,
                src=src,
                alert_id=str(alert_id) if alert_id else None,
                run_id=str(run_id) if run_id else None,
                airline_code=airline_code or None,
                origin=origin,
                destination=destination,
                dep_date=dep_str,
                ret_date=ret_str,
                cabin=cabin,
                passengers=passengers,
                price=float(price_val) if price_val else None,
            )

        return {
            "skyscanner": _track(raw.get("skyscanner"), "skyscanner"),
            "kayak": _track(raw.get("kayak"), "kayak"),
            "airline": _track(raw.get("airline"), "airline"),
        }

    except Exception as e:
        print(f"[alerts_email] booking urls failed: {e}")
        return {"skyscanner": None, "kayak": None, "airline": None}


def _booking_buttons_html(
    booking_urls: Dict[str, Optional[str]],
    airline_name: Optional[str] = None,
) -> str:
    """
    Renders the 3 booking CTA buttons for an email row.
    Skyscanner and Kayak always shown. Airline direct only if URL available.
    """
    airline_label = f"Book with {airline_name}" if airline_name else "Book direct"

    sky_url = booking_urls.get("skyscanner")
    kayak_url = booking_urls.get("kayak")
    airline_url = booking_urls.get("airline")

    buttons = []

    if sky_url:
        buttons.append(
            f'<a href="{sky_url}" style="display:inline-block;background:#0770e3;color:#ffffff;'
            f'text-decoration:none;padding:9px 14px;border-radius:8px;font-weight:700;'
            f'font-size:13px;margin-right:6px;margin-bottom:6px;">Skyscanner ↗</a>'
        )

    if kayak_url:
        buttons.append(
            f'<a href="{kayak_url}" style="display:inline-block;background:#ff690f;color:#ffffff;'
            f'text-decoration:none;padding:9px 14px;border-radius:8px;font-weight:700;'
            f'font-size:13px;margin-right:6px;margin-bottom:6px;">Kayak ↗</a>'
        )

    if airline_url:
        buttons.append(
            f'<a href="{airline_url}" style="display:inline-block;background:#111827;color:#ffffff;'
            f'text-decoration:none;padding:9px 14px;border-radius:8px;font-weight:700;'
            f'font-size:13px;margin-right:6px;margin-bottom:6px;">{airline_label} ↗</a>'
        )

    return "".join(buttons) if buttons else ""

# =====================================================================
# SECTION END: BOOKING URL + TRACKING HELPERS
# =====================================================================


# =====================================================================
# SECTION START: LOW LEVEL UTILS
# =====================================================================

def _smtp_ready() -> bool:
    return bool(SMTP_USERNAME and SMTP_PASSWORD and ALERT_FROM_EMAIL)


def _safe_int(val, default: int = 1) -> int:
    try:
        n = int(val)
        return n if n > 0 else default
    except Exception:
        return default


def _safe_float(val, default: float = 0.0) -> float:
    try:
        return float(val)
    except Exception:
        return default


def _get_attr(obj, name: str, default=None):
    return getattr(obj, name, default)


def _derive_passengers(alert=None, params=None) -> int:
    """
    Priority:
    1) alert.passengers (if present)
    2) params.passengers (if present)
    3) default 1
    """
    if alert is not None:
        p = _get_attr(alert, "passengers", None)
        if p is not None:
            return _safe_int(p, 1)

    if params is not None:
        p = _get_attr(params, "passengers", None)
        if p is not None:
            return _safe_int(p, 1)

    return 1


def _passengers_label(passengers: int) -> str:
    return "1 passenger" if passengers == 1 else f"{passengers} passengers"


def _price_basis_line() -> str:
    """
    Product stance for emails:
    - All prices shown are per passenger
    """
    return "Prices shown are per passenger."


def _is_flex_alert(alert) -> bool:
    """
    Determines if alert represents a flexible window (FlyyvFlex) style alert.
    """
    search_mode = (_get_attr(alert, "search_mode", "") or "").strip().lower()
    mode = (_get_attr(alert, "mode", "") or "").strip().lower()
    return (mode == "smart") or (search_mode == "flexible")


def _compute_trip_nights(alert) -> Optional[int]:
    """
    Attempts to derive a fixed trip length in nights.

    Priority:
    1) return_start - departure_start (if both present)
    2) alert.nights (if present)
    """
    dep_start = _get_attr(alert, "departure_start", None)
    ret_start = _get_attr(alert, "return_start", None)

    if dep_start and ret_start:
        try:
            n = (ret_start - dep_start).days
            return max(1, int(n))
        except Exception:
            return None

    nights = _get_attr(alert, "nights", None)
    if nights is not None:
        n = _safe_int(nights, 0)
        return n if n > 0 else None

    return None


def _compute_theoretical_combinations(alert) -> Optional[int]:
    """
    For FlyyvFlex with fixed nights:
      valid_departures = dep_start .. (dep_end - nights) inclusive
      combos = max(0, (dep_end - dep_start).days - nights + 1)

    Returns None if not computable.
    """
    if not _is_flex_alert(alert):
        return None

    dep_start = _get_attr(alert, "departure_start", None)
    dep_end = _get_attr(alert, "departure_end", None)
    nights = _compute_trip_nights(alert)

    if not (dep_start and dep_end and nights):
        return None

    try:
        total_days = (dep_end - dep_start).days
        valid = total_days - int(nights) + 1
        return max(0, int(valid))
    except Exception:
        return None


def _price_per_pax(total_price, passengers: int) -> float:
    """
    Canonical: email prices are always per passenger.
    We defensively assume incoming 'price' may be total for all passengers.
    """
    pax = _safe_int(passengers, 1)
    if pax <= 0:
        pax = 1
    total = _safe_float(total_price, 0.0)
    return total / float(pax)

# =====================================================================
# SECTION END: LOW LEVEL UTILS
# =====================================================================


# =====================================================================
# SECTION START: GENERIC SINGLE EMAIL SENDER
# =====================================================================

def send_single_email(to_email: str, subject: str, body: str) -> None:
    """
    Generic plain text sender, used for simple operational emails.
    """
    if not _smtp_ready():
        raise HTTPException(status_code=500, detail="SMTP settings are not fully configured on the server")

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = f"FLYYV <{ALERT_FROM_EMAIL}>"
    msg["To"] = to_email
    msg.set_content(body)

    with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
        server.starttls()
        server.login(SMTP_USERNAME, SMTP_PASSWORD)
        server.send_message(msg)

# =====================================================================
# SECTION END: GENERIC SINGLE EMAIL SENDER
# =====================================================================


# =====================================================================
# SECTION START: LINK BUILDERS
# =====================================================================

def build_flyyv_link(
    alert_or_params,
    departure: str,
    return_date: str,
    passengers: Optional[int] = None,
    alert_run_id: Optional[str] = None,
) -> str:
    """
    Deep link to a single date pair drilldown.
    Must use /SearchFlyyv with autoSearch=1 and searchMode=single.

    Optional:
    - alert_run_id: once snapshots exist, FE can render exact run results
    """
    base = FRONTEND_BASE_URL.rstrip("/")

    origin = _get_attr(alert_or_params, "origin", "")
    destination = _get_attr(alert_or_params, "destination", "")
    cabin = _get_attr(alert_or_params, "cabin", "BUSINESS")

    alert_id = _get_attr(alert_or_params, "id", None)
    pax = passengers if passengers is not None else _derive_passengers(alert=alert_or_params, params=alert_or_params)

    qp = {
        "origin": origin,
        "destination": destination,
        "cabin": cabin,
        "passengers": str(_safe_int(pax, 1)),
        "searchMode": "single",
        "departureStart": departure,
        "departureEnd": departure,
        "returnStart": return_date,
        "returnEnd": return_date,
        "autoSearch": "1",
    }

    if alert_id is not None:
        qp["alertId"] = str(alert_id)

    if alert_run_id:
        qp["alertRunId"] = str(alert_run_id)

    return f"{base}/SearchFlyyv?{urlencode(qp)}"


def build_alert_search_link(alert, alert_run_id: Optional[str] = None) -> str:
    """
    Deep link to recreate the original alert window.

    Optional:
    - alert_run_id: once snapshots exist, FE can render exact run results
    """
    base = FRONTEND_BASE_URL.rstrip("/")
    is_flex = _is_flex_alert(alert)
    pax = _derive_passengers(alert=alert, params=None)

    dep_start = _get_attr(alert, "departure_start", None)
    dep_end = _get_attr(alert, "departure_end", None)
    ret_start = _get_attr(alert, "return_start", None)
    ret_end = _get_attr(alert, "return_end", None)

    qp = {
        "origin": _get_attr(alert, "origin", ""),
        "destination": _get_attr(alert, "destination", ""),
        "cabin": _get_attr(alert, "cabin", "BUSINESS"),
        "passengers": str(_safe_int(pax, 1)),
        "searchMode": "flexible" if is_flex else "single",
        "departureStart": dep_start.isoformat() if dep_start else "",
        "departureEnd": dep_end.isoformat() if dep_end else "",
        "autoSearch": "1",
        "alertId": str(_get_attr(alert, "id", "")),
    }

    if ret_start:
        qp["returnStart"] = ret_start.isoformat()
    if ret_end:
        qp["returnEnd"] = ret_end.isoformat()

    nights = _compute_trip_nights(alert)
    if nights:
        qp["nights"] = str(nights)

    if alert_run_id:
        qp["alertRunId"] = str(alert_run_id)

    qp = {k: v for k, v in qp.items() if v not in ("", None)}
    return f"{base}/SearchFlyyv?{urlencode(qp)}"

# =====================================================================
# SECTION END: LINK BUILDERS
# =====================================================================


# =====================================================================
# SECTION START: DISPLAY HELPERS
# =====================================================================

def _cabin_display_label(cabin) -> str:
    """
    User facing cabin label, never show raw enum values like PREMIUM_ECONOMY.
    """
    raw = str(cabin or "").strip()
    raw_upper = raw.upper()

    mapping = {
        "ECONOMY": "Economy",
        "PREMIUM_ECONOMY": "Premium Economy",
        "BUSINESS": "Business Class",
        "FIRST": "First Class",
    }
    if raw_upper in mapping:
        return mapping[raw_upper]

    raw_clean = raw.replace("_", " ").strip()
    return raw_clean.title() if raw_clean else "Business Class"


def _pill_cabin_label(cabin) -> str:
    """
    Cabin label for pills, Title Case, not all caps.
    """
    return _cabin_display_label(cabin)


def _fmt_money_gbp(value) -> str:
    """
    Formats money for display. Current behaviour: integer pounds, no decimals.
    """
    try:
        return f"£{int(float(value))}"
    except Exception:
        return "£0"


def _fmt_date_label(iso_or_dt) -> str:
    """
    Converts ISO string or datetime/date into "DD Mon YYYY".
    """
    try:
        if isinstance(iso_or_dt, str):
            dt = datetime.fromisoformat(iso_or_dt)
        else:
            dt = iso_or_dt
        return dt.strftime("%d %b %Y")
    except Exception:
        return "Unknown date"

# =====================================================================
# SECTION END: DISPLAY HELPERS
# =====================================================================

# =====================================================================
# SECTION START: ONE OFF ALERT EMAIL
# =====================================================================

def send_alert_email_for_alert(alert, cheapest, params, alert_run_id: Optional[str] = None) -> None:
    """
    One off alert email:
    - single date pair
    - single cheapest option
    Prices must be per person.
    """
    if not _smtp_ready():
        raise HTTPException(status_code=500, detail="SMTP settings are not fully configured on the server")

    to_email = _get_attr(alert, "user_email", None)
    if not to_email:
        raise HTTPException(status_code=500, detail="Alert has no user_email")

    origin = _get_attr(alert, "origin", "")
    destination = _get_attr(alert, "destination", "")
    cabin = _get_attr(alert, "cabin", "BUSINESS")

    passengers = _derive_passengers(alert=alert, params=params)
    passenger_text = _passengers_label(passengers)

    # Canonical: per person
    total_price = _get_attr(cheapest, "price", 0)
    per_pax_price = _price_per_pax(total_price, passengers)
    price_label = _fmt_money_gbp(per_pax_price)

    cabin_title = _cabin_display_label(cabin)

    dep_label = _fmt_date_label(getattr(cheapest, "departureDate", None))
    ret_label = _fmt_date_label(getattr(cheapest, "returnDate", None))

    drill_url = build_flyyv_link(
        alert,
        getattr(cheapest, "departureDate", None),
        getattr(cheapest, "returnDate", None),
        passengers=passengers,
        alert_run_id=alert_run_id,
    )

    # Tracked Flyyv link
    flyyv_tracked = _build_tracked_url(
        destination_url=drill_url,
        src="flyyv",
        alert_id=str(_get_attr(alert, "id", "")) or None,
        run_id=str(alert_run_id) if alert_run_id else None,
        airline_code=getattr(cheapest, "airlineCode", None),
        origin=origin,
        destination=destination,
        dep_date=str(getattr(cheapest, "departureDate", ""))[:10] or None,
        ret_date=str(getattr(cheapest, "returnDate", ""))[:10] or None,
        cabin=cabin,
        passengers=passengers,
        price=float(_get_attr(cheapest, "price", 0) or 0),
    )

    # Booking URLs
    booking_urls = _get_booking_urls_for_result(
        cheapest,
        cabin=cabin,
        passengers=passengers,
        alert_id=str(_get_attr(alert, "id", "")),
        run_id=alert_run_id,
    )
    booking_buttons = _booking_buttons_html(booking_urls, airline_label)

    airline_label = getattr(cheapest, "airline", None) or "Multiple airlines"
    airline_code = getattr(cheapest, "airlineCode", None) or ""
    airline_code_txt = f" ({airline_code})" if airline_code else ""

    subject = f"Flyyv Alert: {origin} → {destination} from {price_label} per person"

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = f"FLYYV <{ALERT_FROM_EMAIL}>"
    msg["To"] = to_email

    # ================================================================
    # SECTION START: PLAIN TEXT FALLBACK
    # ================================================================

    lines: List[str] = []
    lines.append("Flyyv Alert")
    lines.append(f"Route: {origin} → {destination}, {cabin_title}")
    lines.append(f"Passengers: {passengers}")
    lines.append(_price_basis_line())
    lines.append(f"Dates: {dep_label} to {ret_label}")
    lines.append("")
    lines.append(f"Best price found: {price_label} per person with {airline_label}{airline_code_txt}")
    lines.append("")
    lines.append("View this flight:")
    lines.append(drill_url)
    lines.append("")
    lines.append("You are receiving this because you created a Flyyv price alert.")
    lines.append("To stop these alerts, delete the alert in your Flyyv profile.")
    msg.set_content("\n".join(lines))

    # ================================================================
    # SECTION END: PLAIN TEXT FALLBACK
    # ================================================================

    # ================================================================
    # SECTION START: HTML EMAIL TEMPLATE
    # ================================================================

    html = f"""
    <html>
      <body style="margin:0;padding:0;background:#f6f7f9;font-family:Arial,Helvetica,sans-serif;">
        <div style="max-width:680px;margin:0 auto;padding:24px;">

          <div style="background:#ffffff;border:1px solid #e6e8ee;border-radius:14px;padding:26px;">

            <div style="font-size:14px;color:#6b7280;margin-bottom:10px;">Flyyv Alert</div>

            <div style="font-size:28px;line-height:1.2;color:#111827;font-weight:800;margin:0 0 10px 0;">
              Price match found
            </div>

            <div style="font-size:16px;line-height:1.5;color:#111827;margin:0 0 14px 0;">
              {origin} → {destination}, <strong>{cabin_title}</strong>, {passenger_text}
            </div>

            <div style="border:1px solid #e6e8ee;border-radius:14px;padding:16px;background:#fbfbfd;margin:0 0 18px 0;">
              <div style="font-size:22px;color:#111827;font-weight:900;line-height:1.1;margin:0 0 4px 0;">
                {price_label}
              </div>
              <div style="font-size:13px;color:#6b7280;margin:0 0 10px 0;">
                per person, {dep_label} to {ret_label}
              </div>
              <div style="font-size:14px;color:#111827;margin:0 0 14px 0;">
                Cheapest option: <strong>{airline_label}</strong>{airline_code_txt}
              </div>
              <div style="margin-bottom:4px;">
                {booking_buttons}
              </div>
            </div>

            <div style="font-size:12px;color:#9ca3af;margin:0 0 16px 0;">
              Prices change fast. Flyyv never marks up fares — you book directly with the provider.
            </div>

            <div style="margin:0 0 18px 0;">
              <a href="{flyyv_tracked}"
                 style="font-size:13px;color:#6b7280;text-decoration:underline;">
                View results on Flyyv
              </a>
            </div>

            <div style="font-size:12px;color:#6b7280;line-height:1.4;">
              You are receiving this email because you created a Flyyv price alert.
              To stop these alerts, delete the alert in your Flyyv profile.
            </div>

          </div>

        </div>
      </body>
    </html>
    """

    # ================================================================
    # SECTION END: HTML EMAIL TEMPLATE
    # ================================================================

    msg.add_alternative(html, subtype="html")

    with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
        server.starttls()
        server.login(SMTP_USERNAME, SMTP_PASSWORD)
        server.send_message(msg)

# =====================================================================
# SECTION END: ONE OFF ALERT EMAIL
# =====================================================================


# =====================================================================
# SECTION START: SMART ALERT SUMMARY EMAIL (FLYYVFLEX)
# =====================================================================

def send_smart_alert_email(alert, options: List, params, alert_run_id: Optional[str] = None) -> None:
    """
    FlyyvFlex results email:
    - Top 5 cheapest date combinations
    - Per-row CTA drills into a single date pair
    - Full results CTA recreates original window
    Prices must be per person.
    """
    if not _smtp_ready():
        raise HTTPException(status_code=500, detail="SMTP settings are not fully configured on the server")

    to_email = _get_attr(alert, "user_email", None)
    if not to_email:
        raise HTTPException(status_code=500, detail="Alert has no user_email")

    threshold = _get_attr(alert, "max_price", None)
    origin = _get_attr(alert, "origin", "")
    destination = _get_attr(alert, "destination", "")
    cabin = _get_attr(alert, "cabin", "BUSINESS")

    passengers = _derive_passengers(alert=alert, params=params)
    passenger_text = _passengers_label(passengers)

    cabin_title = _cabin_display_label(cabin)
    cabin_pill = _pill_cabin_label(cabin)

    # ================================================================
    # SECTION START: GROUP OPTIONS BY DATE PAIR
    # ================================================================

    grouped: Dict[Tuple[str, str], List] = {}
    for opt in options:
        key = (opt.departureDate, opt.returnDate)
        grouped.setdefault(key, []).append(opt)

    # ================================================================
    # SECTION END: GROUP OPTIONS BY DATE PAIR
    # ================================================================

    any_under = False
    pairs_summary: List[Dict[str, Any]] = []

    # ================================================================
    # SECTION START: BUILD PER-PAIR SUMMARY
    # ================================================================

    for dep_iso, ret_iso in grouped.keys():
        flights = grouped[(dep_iso, ret_iso)]
        totals = [o.price for o in flights if getattr(o, "price", None) is not None]
        if not totals:
            continue

        cheapest = min(flights, key=lambda o: o.price)
        min_total_price = float(min(totals))

        # Canonical: per person
        cheapest_per_pax = _price_per_pax(getattr(cheapest, "price", 0), passengers)
        min_per_pax = _price_per_pax(min_total_price, passengers)

        flights_under: List = []
        if threshold is not None:
            try:
                # Interpret threshold as per person
                th = float(threshold)
                flights_under = [o for o in flights if _price_per_pax(o.price, passengers) <= th]
                if flights_under:
                    any_under = True
            except Exception:
                flights_under = []

        flyyv_link = build_flyyv_link(alert, dep_iso, ret_iso, passengers=passengers, alert_run_id=alert_run_id)

        pairs_summary.append(
            {
                "departureDate": dep_iso,
                "returnDate": ret_iso,
                "totalFlights": len(flights),
                "cheapestPrice": float(cheapest_per_pax),  # per person
                "cheapestAirline": getattr(cheapest, "airline", None),
                "cheapestAirlineCode": getattr(cheapest, "airlineCode", None),
                "cheapestResult": cheapest,  # kept for booking URL generation
                "flyyvLink": flyyv_link,
                "minPrice": float(min_per_pax),  # per person
                "flightsUnderThresholdCount": len(flights_under),
            }
        )

    # ================================================================
    # SECTION END: BUILD PER-PAIR SUMMARY
    # ================================================================

    start_label = params.earliestDeparture.strftime("%d %b %Y")
    end_label = params.latestDeparture.strftime("%d %b %Y")

    nights_val = _compute_trip_nights(alert)
    nights_text = str(nights_val) if nights_val else None

    theoretical_combinations = _compute_theoretical_combinations(alert)
    analysed_combinations = theoretical_combinations if theoretical_combinations is not None else len(pairs_summary)

    best_price_overall: Optional[int] = None
    if pairs_summary:
        try:
            best_price_overall = int(min(pairs_summary, key=lambda x: x["cheapestPrice"])["cheapestPrice"])
        except Exception:
            best_price_overall = None

    # ================================================================
    # SECTION START: SUBJECT
    # ================================================================

    if best_price_overall is not None:
        subject = f"FlyyvFlex Alert: {origin} → {destination} from £{best_price_overall} per person"
    elif threshold is not None and any_under:
        try:
            subject = f"FlyyvFlex Alert: {origin} → {destination} fares under £{int(float(threshold))} per person"
        except Exception:
            subject = f"FlyyvFlex Alert: {origin} → {destination} fares under your price range"
    else:
        subject = f"FlyyvFlex Alert: {origin} → {destination} update"

    # ================================================================
    # SECTION END: SUBJECT
    # ================================================================

    top_pairs = [p for p in pairs_summary if p.get("cheapestPrice") is not None]
    top_pairs_sorted = sorted(top_pairs, key=lambda x: x["cheapestPrice"])[:5]

    open_full_results_url = build_alert_search_link(alert, alert_run_id=alert_run_id)
    open_full_results_tracked = _build_tracked_url(
        destination_url=open_full_results_url,
        src="flyyv_full",
        alert_id=str(_get_attr(alert, "id", "")) or None,
        run_id=alert_run_id,
        origin=origin,
        destination=destination,
        cabin=cabin,
        passengers=passengers,
    )

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = f"FLYYV <{ALERT_FROM_EMAIL}>"
    msg["To"] = to_email

    # ================================================================
    # SECTION START: PLAIN TEXT FALLBACK
    # ================================================================

    lines: List[str] = []
    lines.append("FlyyvFlex Smart Search Alert")
    lines.append(f"Route: {origin} → {destination}, {cabin_title}")
    lines.append(f"Passengers: {passengers}")
    lines.append(_price_basis_line())
    if nights_text:
        lines.append(f"Trip length: {nights_text} nights")
    lines.append(f"Date window scanned: {start_label} to {end_label}")
    lines.append(f"Date combinations analysed: {analysed_combinations}")

    if best_price_overall is not None:
        lines.append(f"Best price found: £{best_price_overall} per person")

    lines.append("")
    lines.append("Top 5 cheapest date combinations:")
    lines.append("")

    if not top_pairs_sorted:
        lines.append("No flights were found in this window in the latest scan.")
    else:
        for p in top_pairs_sorted:
            dep_label = _fmt_date_label(p["departureDate"])
            ret_label = _fmt_date_label(p["returnDate"])
            price_label = _fmt_money_gbp(p["cheapestPrice"])
            airline_label = p.get("cheapestAirline") or "Multiple airlines"

            within_txt = ""
            try:
                if threshold is not None and float(p["cheapestPrice"]) <= float(threshold):
                    within_txt = f", within your £{int(float(threshold))} price range"
            except Exception:
                within_txt = ""

            lines.append(f"{price_label} per person | {dep_label} to {ret_label} | {airline_label}{within_txt}")
            lines.append(f"View flight: {p.get('flyyvLink')}")
            lines.append("")

    lines.append("Open full results:")
    lines.append(open_full_results_url)
    msg.set_content("\n".join(lines))

    # ================================================================
    # SECTION END: PLAIN TEXT FALLBACK
    # ================================================================

    # ================================================================
    # SECTION START: TOP 5 ROWS HTML
    # ================================================================

    rows_html = ""
    alert_id_str = str(_get_attr(alert, "id", "")) or None
    for p in top_pairs_sorted:
        dep_label = _fmt_date_label(p["departureDate"])
        ret_label = _fmt_date_label(p["returnDate"])

        price_label = _fmt_money_gbp(p["cheapestPrice"])
        airline_label = p.get("cheapestAirline") or "Multiple airlines"
        flyyv_link = p.get("flyyvLink") or open_full_results_url

        # Tracked Flyyv link for this row
        flyyv_row_tracked = _build_tracked_url(
            destination_url=flyyv_link,
            src="flyyv",
            alert_id=alert_id_str,
            run_id=alert_run_id,
            airline_code=p.get("cheapestAirlineCode"),
            origin=origin,
            destination=destination,
            dep_date=p["departureDate"][:10],
            ret_date=p["returnDate"][:10],
            cabin=cabin,
            passengers=passengers,
            price=p.get("cheapestPrice"),
        )

        # Build booking buttons for this row
        cheapest_result = p.get("cheapestResult")
        if cheapest_result is not None:
            row_booking_urls = _get_booking_urls_for_result(
                cheapest_result,
                cabin=cabin,
                passengers=passengers,
                alert_id=alert_id_str,
                run_id=alert_run_id,
            )
        else:
            row_booking_urls = {"skyscanner": None, "kayak": None, "airline": None}

        row_buttons = _booking_buttons_html(row_booking_urls, airline_label if p.get("cheapestAirlineCode") else None)

        within = False
        threshold_int = None
        try:
            if threshold is not None:
                threshold_int = int(float(threshold))
                if float(p["cheapestPrice"]) <= float(threshold):
                    within = True
        except Exception:
            within = False
            threshold_int = None

        within_html = ""
        if within and threshold_int is not None:
            within_html = f'<span style="color:#059669;font-weight:700;"> ✓ Within your £{threshold_int} budget</span>'

        rows_html += f"""
          <tr>
            <td style="padding:0 0 12px 0;">
              <table role="presentation" width="100%" cellpadding="0" cellspacing="0"
                     style="border:1px solid #e6e8ee;border-radius:12px;background:#ffffff;border-collapse:separate;">
                <tr>
                  <td style="padding:14px 14px 6px 14px;vertical-align:top;">
                    <div style="display:table;width:100%;">
                      <div style="display:table-cell;vertical-align:top;">
                        <div style="font-size:14px;color:#111827;font-weight:700;margin-bottom:3px;">
                          {origin} → {destination} &nbsp;·&nbsp; {dep_label} → {ret_label}
                        </div>
                        <div style="font-size:13px;color:#6b7280;margin-bottom:6px;">
                          From: <strong>{airline_label}</strong>{within_html}
                        </div>
                      </div>
                      <div style="display:table-cell;vertical-align:top;text-align:right;white-space:nowrap;padding-left:12px;">
                        <div style="font-size:20px;color:#111827;font-weight:900;line-height:1.1;">{price_label}</div>
                        <div style="font-size:11px;color:#6b7280;margin-top:2px;">per person</div>
                      </div>
                    </div>
                  </td>
                </tr>
                <tr>
                  <td style="padding:0 14px 14px 14px;">
                    {row_buttons}
                    <a href="{flyyv_row_tracked}" style="display:inline-block;font-size:12px;color:#9ca3af;text-decoration:underline;margin-top:4px;">
                      View on Flyyv
                    </a>
                  </td>
                </tr>
              </table>
            </td>
          </tr>
        """

    # ================================================================
    # SECTION END: TOP 5 ROWS HTML
    # ================================================================

    # ================================================================
    # SECTION START: CHIPS
    # ================================================================

    best_chip = ""
    if best_price_overall is not None:
        best_chip = f"""
        <span style="display:inline-block;padding:8px 12px;border-radius:999px;border:1px solid #e6e8ee;background:#f9fafb;font-size:13px;margin-bottom:8px;">
          Best {_fmt_money_gbp(best_price_overall)} per person
        </span>
        """

    combos_chip = f"""
    <span style="display:inline-block;padding:8px 12px;border-radius:999px;border:1px solid #e6e8ee;background:#f9fafb;font-size:13px;margin-right:8px;margin-bottom:8px;">
      {int(analysed_combinations)} date combinations analysed
    </span>
    """

    # ================================================================
    # SECTION END: CHIPS
    # ================================================================

    # ================================================================
    # SECTION START: HTML EMAIL TEMPLATE
    # ================================================================

    html = f"""
    <html>
      <body style="margin:0;padding:0;background:#f6f7f9;font-family:Arial,Helvetica,sans-serif;">
        <div style="max-width:680px;margin:0 auto;padding:24px;">

          <div style="background:#ffffff;border:1px solid #e6e8ee;border-radius:14px;padding:26px;">

            <div style="font-size:14px;color:#6b7280;margin-bottom:10px;">FlyyvFlex Smart Search Alert</div>

            <div style="font-size:28px;line-height:1.2;color:#111827;font-weight:900;margin:0 0 10px 0;">
              Top {cabin_title} deals for {passenger_text} going from {origin} → {destination}
            </div>

            <div style="font-size:15px;line-height:1.6;color:#111827;margin:0 0 14px 0;">
              Based on a scan of your <strong>{start_label} to {end_label}</strong> window
              {f" for <strong>{nights_text}-night</strong> trips" if nights_text else ""}.
            </div>

            <div style="font-size:13px;color:#6b7280;margin:0 0 12px 0;">
              Passengers: <strong>{passenger_text}</strong><br>
              {_price_basis_line()}
            </div>

            <div style="margin:0 0 16px 0;">

              <span style="display:inline-block;padding:8px 12px;border-radius:999px;border:1px solid #d1fae5;background:#ecfdf5;font-size:13px;font-weight:800;margin-right:8px;margin-bottom:8px;">
                {cabin_pill}
              </span>

              <span style="display:inline-block;padding:8px 12px;border-radius:999px;border:1px solid #e6e8ee;background:#f9fafb;font-size:13px;margin-right:8px;margin-bottom:8px;">
                {passenger_text}
              </span>

              {combos_chip}
              {best_chip}
            </div>

            <div style="border-top:1px solid #eef0f5;margin:14px 0;"></div>

            <div style="font-size:18px;color:#111827;font-weight:900;margin:0 0 12px 0;">
              Top 5 cheapest date combinations
            </div>

            <table role="presentation" width="100%" cellpadding="0" cellspacing="0" style="border-collapse:separate;">
              {rows_html if rows_html else '<tr><td style="color:#6b7280;font-size:14px;">No flights found in this scan.</td></tr>'}
            </table>

            <div style="margin:18px 0 0 0;">
              <a href="{open_full_results_tracked}"
                 style="display:inline-block;background:#111827;color:#ffffff;text-decoration:none;padding:12px 16px;border-radius:10px;font-weight:900;font-size:15px;">
                Open full results on Flyyv
              </a>
            </div>

            <div style="font-size:12px;color:#6b7280;line-height:1.4;margin-top:14px;">
              You are receiving this email because you created a FlyyvFlex Smart Search Alert.
              To stop these alerts, delete the alert in your Flyyv profile.
            </div>

          </div>

          <div style="text-align:center;font-size:11px;color:#9ca3af;padding:14px 0;">
            Flyyv, <a href="{open_full_results_url}" style="color:#6b7280;text-decoration:underline;">Open your results</a>
          </div>

        </div>
      </body>
    </html>
    """

    # ================================================================
    # SECTION END: HTML EMAIL TEMPLATE
    # ================================================================

    msg.add_alternative(html, subtype="html")

    with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
        server.starttls()
        server.login(SMTP_USERNAME, SMTP_PASSWORD)
        server.send_message(msg)

# =====================================================================
# SECTION END: SMART ALERT SUMMARY EMAIL (FLYYVFLEX)
# =====================================================================


# =====================================================================
# SECTION START: ALERT CONFIRMATION EMAIL
# =====================================================================

def send_alert_confirmation_email(alert) -> None:
    """
    Sent immediately when a user creates an alert.
    Confirms the alert is active.
    """
    if not _smtp_ready():
        return

    to_email = _get_attr(alert, "user_email", None)
    if not to_email:
        return

    origin = _get_attr(alert, "origin", "")
    destination = _get_attr(alert, "destination", "")
    cabin = _get_attr(alert, "cabin", "BUSINESS")
    passengers = _derive_passengers(alert=alert, params=None)

    departure_start = _get_attr(alert, "departure_start", None)
    departure_end = _get_attr(alert, "departure_end", None)

    is_flex = _is_flex_alert(alert)
    email_type_label = "FlyyvFlex Smart Search Alert" if is_flex else "Flyyv Alert"
    pill_type_label = "Smart price watch" if is_flex else "Price alert"

    dep_start_label = departure_start.strftime("%d %b %Y") if departure_start else "Not set"
    dep_end_label = departure_end.strftime("%d %b %Y") if departure_end else "Not set"
    dep_window_label = f"{dep_start_label} to {dep_end_label}"

    nights_val = _compute_trip_nights(alert)
    trip_length_label = f"{nights_val} nights" if nights_val else ("Flexible" if is_flex else "Not set")

    theoretical_combinations = _compute_theoretical_combinations(alert)

    results_url = build_alert_search_link(alert)
    alert_id = _get_attr(alert, "id", None)

    cabin_title = _cabin_display_label(cabin)
    cabin_pill = _pill_cabin_label(cabin)

    subject = f"{email_type_label}: {origin} → {destination} | {dep_start_label} to {dep_end_label} | {trip_length_label}"

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = f"FLYYV <{ALERT_FROM_EMAIL}>"
    msg["To"] = to_email

    # ================================================================
    # SECTION START: PLAIN TEXT FALLBACK
    # ================================================================

    text_body = (
        f"{email_type_label}\n\n"
        "Your alert is active.\n\n"
        f"Route: {origin} → {destination}\n"
        f"Cabin: {cabin_title}\n"
        f"Passengers: {passengers}\n"
        f"{_price_basis_line()}\n"
        f"Departure window: {dep_window_label}\n"
        f"Trip length: {trip_length_label}\n"
        + (f"Combinations in your window: {theoretical_combinations}\n" if theoretical_combinations else "")
        + (f"Alert ID: {alert_id}\n" if alert_id else "")
        + "\nView results:\n"
        f"{results_url}\n"
    )
    msg.set_content(text_body)

    # ================================================================
    # SECTION END: PLAIN TEXT FALLBACK
    # ================================================================

    # ================================================================
    # SECTION START: HTML EMAIL TEMPLATE
    # ================================================================

    html = f"""
    <html>
      <body style="margin:0;padding:0;background:#f6f7f9;font-family:Arial,Helvetica,sans-serif;">
        <div style="max-width:680px;margin:0 auto;padding:24px;">

          <div style="background:#ffffff;border:1px solid #e6e8ee;border-radius:14px;padding:26px;">

            <div style="font-size:14px;color:#6b7280;margin-bottom:10px;">{email_type_label}</div>

            <div style="font-size:28px;line-height:1.2;color:#111827;font-weight:800;margin:0 0 12px 0;">
              Your alert is <span style="color:#059669;font-weight:900;">active</span>
            </div>

            <div style="font-size:16px;line-height:1.5;color:#111827;margin:0 0 12px 0;">
              We are watching <strong>{origin} → {destination}</strong> and will email you when prices match your alert conditions.
            </div>

            <div style="font-size:13px;color:#6b7280;margin:0 0 16px 0;">
              Passengers: <strong>{_passengers_label(passengers)}</strong><br>
              {_price_basis_line()}
            </div>

            <div style="margin:0 0 18px 0;">

              <span style="display:inline-block;padding:8px 12px;border-radius:999px;border:1px solid #e6e8ee;background:#f9fafb;font-size:13px;margin-right:8px;margin-bottom:8px;">
                {origin} → {destination}
              </span>

              <span style="display:inline-block;padding:8px 12px;border-radius:999px;border:1px solid #d1fae5;background:#ecfdf5;font-size:13px;font-weight:700;margin-right:8px;margin-bottom:8px;">
                {cabin_pill}
              </span>

              <span style="display:inline-block;padding:8px 12px;border-radius:999px;border:1px solid #e6e8ee;background:#f9fafb;font-size:13px;margin-right:8px;margin-bottom:8px;">
                {_passengers_label(passengers)}
              </span>

              <span style="display:inline-block;padding:8px 12px;border-radius:999px;border:1px solid #e6e8ee;background:#f9fafb;font-size:13px;margin-right:8px;margin-bottom:8px;">
                {pill_type_label}
              </span>

              <span style="display:inline-block;padding:8px 12px;border-radius:999px;border:1px solid #e6e8ee;background:#f9fafb;font-size:13px;margin-right:8px;margin-bottom:8px;">
                {dep_window_label}
              </span>

              <span style="display:inline-block;padding:8px 12px;border-radius:999px;border:1px solid #e6e8ee;background:#f9fafb;font-size:13px;margin-right:8px;margin-bottom:8px;">
                {trip_length_label}
              </span>

              {f'''
              <span style="display:inline-block;padding:8px 12px;border-radius:999px;border:1px solid #e6e8ee;background:#f9fafb;font-size:13px;margin-bottom:8px;">
                {int(theoretical_combinations)} combinations
              </span>
              ''' if theoretical_combinations else ''}

            </div>

            {f'''
            <div style="font-size:12px;color:#6b7280;margin:0 0 12px 0;">
              Alert ID: {alert_id}
            </div>
            ''' if alert_id else ''}

            <div style="border:1px solid #e6e8ee;border-radius:14px;padding:16px;margin:0 0 18px 0;background:#fbfbfd;">
              <div style="font-size:13px;color:#6b7280;margin-bottom:8px;">What we will do</div>
              <ul style="margin:0;padding-left:18px;color:#111827;font-size:15px;line-height:1.6;">
                <li>Scan your selected window for standout prices</li>
                <li>Notify you when we see meaningful drops or strong value</li>
                <li>Let you jump straight back into your results anytime</li>
              </ul>
            </div>

            <div style="margin:0 0 18px 0;">
              <a href="{results_url}"
                 style="display:inline-block;background:#111827;color:#ffffff;text-decoration:none;padding:12px 16px;border-radius:10px;font-weight:800;font-size:15px;">
                View results
              </a>
            </div>

            <div style="font-size:12px;color:#6b7280;line-height:1.4;">
              You are receiving this email because you created a {email_type_label}.
            </div>

          </div>

          <div style="text-align:center;font-size:11px;color:#9ca3af;padding:14px 0;">
            Flyyv, <a href="{results_url}" style="color:#6b7280;text-decoration:underline;">Open your results</a>
          </div>

        </div>
      </body>
    </html>
    """

    # ================================================================
    # SECTION END: HTML EMAIL TEMPLATE
    # ================================================================

    msg.add_alternative(html, subtype="html")

    try:
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
            server.starttls()
            server.login(SMTP_USERNAME, SMTP_PASSWORD)
            server.send_message(msg)
    except Exception:
        pass

# =====================================================================
# SECTION END: ALERT CONFIRMATION EMAIL
# =====================================================================

# =====================================================================
# SECTION START: EARLY ACCESS WELCOME EMAIL
# =====================================================================

def send_early_access_welcome_email(to_email: str) -> None:
    subject = "Welcome aboard Flyyv"
    body = (
        "Hi there,\n\n"
        "Daniel here, founder of Flyyv.\n\n"
        "Thank you for signing up for early access. "
        "You will be among the very first to try our new platform dedicated to finding exceptional business fares.\n\n"
        "I really appreciate you joining us this early.\n\n"
        "Talk soon,\n"
        "Daniel\n"
        "Founder, Flyyv"
    )

    send_single_email(to_email=to_email, subject=subject, body=body)

# =====================================================================
# SECTION END: EARLY ACCESS WELCOME EMAIL
# =====================================================================