import os
import smtplib
from email.message import EmailMessage
from datetime import datetime, date, timedelta
from typing import List, Dict, Tuple, Any, Optional

from fastapi import HTTPException
from urllib.parse import urlencode

# =====================================================================
# SECTION START: SMTP CONFIG AND CONSTANTS
# =====================================================================

SMTP_HOST = os.environ.get("SMTP_HOST", "smtp.smtp2go.com")
SMTP_PORT = int(os.environ.get("SMTP_PORT", "587"))
SMTP_USERNAME = os.environ.get("SMTP_USERNAME")
SMTP_PASSWORD = os.environ.get("SMTP_PASSWORD")
ALERT_FROM_EMAIL = os.environ.get("ALERT_FROM_EMAIL", "alert@flyyv.com")

FRONTEND_BASE_URL = os.environ.get("FRONTEND_BASE_URL", "https://flyyv.com")

# =====================================================================
# SECTION END: SMTP CONFIG AND CONSTANTS
# =====================================================================


# =====================================================================
# SECTION START: SMALL UTILITIES
# =====================================================================

def _smtp_ready() -> bool:
    return bool(SMTP_USERNAME and SMTP_PASSWORD and ALERT_FROM_EMAIL)


def _safe_int(val, default: int = 1) -> int:
    try:
        n = int(val)
        return n if n > 0 else default
    except Exception:
        return default


def _get_attr(obj, name: str, default=None):
    return getattr(obj, name, default)


def _derive_passengers(alert=None, params=None) -> int:
    """
    Priority:
    1) alert.passengers (if present in DB model)
    2) params.passengers (if SearchParams used)
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


def _price_basis_line(passengers: int) -> str:
    # Duffel total_amount is per passenger in many cases, but your UI is treating shown
    # prices as totals. Keep your current product stance consistent:
    # "total for all passengers"
    return "Prices shown are the total for all passengers."


def _is_flex_alert(alert) -> bool:
    search_mode = (_get_attr(alert, "search_mode", "") or "").strip().lower()
    mode = (_get_attr(alert, "mode", "") or "").strip().lower()
    return (mode == "smart") or (search_mode == "flexible")


def _compute_trip_nights(alert) -> Optional[int]:
    """
    Returns fixed nights if the alert implies a fixed trip length, else None.
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
        return _safe_int(nights, 0) or None

    return None


def _compute_theoretical_combinations(alert) -> Optional[int]:
    """
    For flexible windows:
    - If fixed nights, each departure day in [departure_start..departure_end] is one combination.
    - If not fixed nights, we cannot compute reliably here, return None.
    """
    if not _is_flex_alert(alert):
        return None

    dep_start = _get_attr(alert, "departure_start", None)
    dep_end = _get_attr(alert, "departure_end", None)
    nights = _compute_trip_nights(alert)

    if not (dep_start and dep_end and nights):
        return None

    try:
        days = (dep_end - dep_start).days + 1
        return max(1, int(days))
    except Exception:
        return None

# =====================================================================
# SECTION END: SMALL UTILITIES
# =====================================================================


# =====================================================================
# SECTION START: GENERIC SINGLE EMAIL SENDER
# =====================================================================

def send_single_email(to_email: str, subject: str, body: str) -> None:
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
# SECTION START: HELPER LINK BUILDERS
# =====================================================================

def build_flyyv_link(alert_or_params, departure: str, return_date: str, passengers: Optional[int] = None) -> str:
    """
    Deep link to a single date pair drilldown.
    Must use /SearchFlyyv with autoSearch=1 and searchMode=single.
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

    return f"{base}/SearchFlyyv?{urlencode(qp)}"


def build_alert_search_link(alert) -> str:
    """
    Deep link to recreate the original alert window.
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

    # Include return window if available (flex alerts often use this to compute nights on FE)
    if ret_start:
        qp["returnStart"] = ret_start.isoformat()
    if ret_end:
        qp["returnEnd"] = ret_end.isoformat()

    # Include nights if we can derive it
    nights = _compute_trip_nights(alert)
    if nights:
        qp["nights"] = str(nights)

    # Drop empty values
    qp = {k: v for k, v in qp.items() if v not in ("", None)}

    return f"{base}/SearchFlyyv?{urlencode(qp)}"

# =====================================================================
# SECTION END: HELPER LINK BUILDERS
# =====================================================================


# =====================================================================
# SECTION START: ONE OFF ALERT EMAIL
# =====================================================================

def send_alert_email_for_alert(alert, cheapest, params) -> None:
    """
    One off alert email:
    single date pair, simple format.
    """
    if not _smtp_ready():
        raise HTTPException(status_code=500, detail="SMTP settings are not fully configured on the server")

    to_email = _get_attr(alert, "user_email", None)
    if not to_email:
        raise HTTPException(status_code=500, detail="Alert has no user_email")

    passengers = _derive_passengers(alert=alert, params=params)

    subject = f"Flyyv Alert: {alert.origin} \u2192 {alert.destination} from £{int(cheapest.price)}"

    dep_dt = datetime.fromisoformat(cheapest.departureDate)
    ret_dt = datetime.fromisoformat(cheapest.returnDate)
    dep_label = dep_dt.strftime("%d %b %Y")
    ret_label = ret_dt.strftime("%d %b %Y")

    drill_url = build_flyyv_link(alert, cheapest.departureDate, cheapest.returnDate, passengers=passengers)

    lines: List[str] = []
    lines.append(f"Route: {alert.origin} \u2192 {alert.destination}, {alert.cabin.title()} class")
    lines.append(f"Passengers: {passengers}")
    lines.append(_price_basis_line(passengers))
    lines.append(f"Dates: {dep_label} to {ret_label}")
    lines.append("")
    lines.append(f"Best price found: £{int(cheapest.price)} with {cheapest.airline} ({cheapest.airlineCode or ''})")
    lines.append("")
    lines.append("View this flight:")
    lines.append(drill_url)
    lines.append("")
    lines.append("You are receiving this because you created a Flyyv price alert.")
    lines.append("To stop these alerts, delete the alert in your Flyyv profile.")

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = f"FLYYV <{ALERT_FROM_EMAIL}>"
    msg["To"] = to_email
    msg.set_content("\n".join(lines))

    with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
        server.starttls()
        server.login(SMTP_USERNAME, SMTP_PASSWORD)
        server.send_message(msg)

# =====================================================================
# SECTION END: ONE OFF ALERT EMAIL
# =====================================================================


# =====================================================================
# SECTION START: SMART ALERT SUMMARY EMAIL
# =====================================================================

def send_smart_alert_email(alert, options: List, params) -> None:
    """
    FlyyvFlex results email:
    summary of multiple date pairs in a flexible window.

    Output:
    - Top 5 cheapest date combinations
    - Per-row CTA drills into single date pair results
    - Full results CTA recreates original window
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

    grouped: Dict[Tuple[str, str], List] = {}
    for opt in options:
        key = (opt.departureDate, opt.returnDate)
        grouped.setdefault(key, []).append(opt)

    any_under = False
    pairs_summary: List[Dict[str, Any]] = []

    for dep_iso, ret_iso in grouped.keys():
        flights = grouped[(dep_iso, ret_iso)]
        prices = [o.price for o in flights if getattr(o, "price", None) is not None]
        if not prices:
            continue

        cheapest = min(flights, key=lambda o: o.price)
        min_price = float(min(prices))

        flights_under: List = []
        if threshold is not None:
            try:
                flights_under = [o for o in flights if o.price <= float(threshold)]
                if flights_under:
                    any_under = True
            except Exception:
                flights_under = []

        flyyv_link = build_flyyv_link(alert, dep_iso, ret_iso, passengers=passengers)

        pairs_summary.append(
            {
                "departureDate": dep_iso,
                "returnDate": ret_iso,
                "totalFlights": len(flights),
                "cheapestPrice": float(cheapest.price),
                "cheapestAirline": getattr(cheapest, "airline", None),
                "flyyvLink": flyyv_link,
                "minPrice": min_price,
                "flightsUnderThresholdCount": len(flights_under),
            }
        )

    start_label = params.earliestDeparture.strftime("%d %b %Y")
    end_label = params.latestDeparture.strftime("%d %b %Y")

    nights_val = _compute_trip_nights(alert)
    nights_text = str(nights_val) if nights_val else None

    analysed_combinations = len(pairs_summary)

    best_price_overall = None
    if pairs_summary:
        try:
            best_price_overall = int(min(pairs_summary, key=lambda x: x["cheapestPrice"])["cheapestPrice"])
        except Exception:
            best_price_overall = None

    if best_price_overall is not None:
        subject = f"FlyyvFlex Alert: {origin} → {destination} from £{best_price_overall} per passenger"
    elif threshold is not None and any_under:
        subject = f"FlyyvFlex Alert: {origin} → {destination} fares under £{int(threshold)} per passenger"
    else:
        subject = f"FlyyvFlex Alert: {origin} → {destination} update"

    top_pairs = [p for p in pairs_summary if p.get("cheapestPrice") is not None]
    top_pairs_sorted = sorted(top_pairs, key=lambda x: x["cheapestPrice"])[:5]

    open_full_results_url = build_alert_search_link(alert)

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = f"FLYYV <{ALERT_FROM_EMAIL}>"
    msg["To"] = to_email

    # Plain text fallback
    lines: List[str] = []
    lines.append("FlyyvFlex Smart Search Alert")
    lines.append(f"Route: {origin} → {destination}, {str(cabin).title()} class")
    lines.append(f"Passengers: {passengers}")
    lines.append("Prices shown are per passenger")
    if nights_text:
        lines.append(f"Trip length: {nights_text} nights")
    lines.append(f"Date window scanned: {start_label} to {end_label}")
    lines.append(f"Date combinations analysed: {analysed_combinations}")

    if best_price_overall is not None:
        lines.append(f"Best price found: £{best_price_overall} per passenger")

    lines.append("")
    lines.append("Top 5 cheapest date combinations:")
    lines.append("")

    if not top_pairs_sorted:
        lines.append("No flights were found in this window in the latest scan.")
    else:
        for p in top_pairs_sorted:
            dep_dt = datetime.fromisoformat(p["departureDate"])
            ret_dt = datetime.fromisoformat(p["returnDate"])
            dep_label = dep_dt.strftime("%d %b %Y")
            ret_label = ret_dt.strftime("%d %b %Y")
            price_label = int(p["cheapestPrice"])
            airline_label = p.get("cheapestAirline") or "Multiple airlines"
            lines.append(f"£{price_label} per passenger | {dep_label} to {ret_label} | {airline_label}")
            lines.append(f"View flight: {p.get('flyyvLink')}")
            lines.append("")

    lines.append("Open full results:")
    lines.append(open_full_results_url)
    msg.set_content("\n".join(lines))

    # HTML
    rows_html = ""
    for p in top_pairs_sorted:
        dep_dt = datetime.fromisoformat(p["departureDate"])
        ret_dt = datetime.fromisoformat(p["returnDate"])
        dep_label = dep_dt.strftime("%d %b %Y")
        ret_label = ret_dt.strftime("%d %b %Y")

        price_label = int(p["cheapestPrice"])
        airline_label = p.get("cheapestAirline") or "Multiple airlines"
        view_link = p.get("flyyvLink") or open_full_results_url

        within = False
        try:
            if threshold is not None and float(price_label) <= float(threshold):
                within = True
        except Exception:
            within = False

        rows_html += f"""
          <tr>
            <td style="padding:14px 14px;border:1px solid #e6e8ee;border-radius:12px;background:#ffffff;display:block;margin-bottom:10px;">
              <div style="display:flex;justify-content:space-between;align-items:flex-start;gap:12px;">
                <div>
                  <div style="font-size:14px;color:#111827;font-weight:700;margin-bottom:4px;">
                    {origin} → {destination}
                  </div>
                  <div style="font-size:13px;color:#6b7280;margin-bottom:6px;">
                    {dep_label} to {ret_label}
                  </div>
                  <div style="font-size:13px;color:#111827;">
                    Cheapest option: <strong>{airline_label}</strong>
                    {('<span style="color:#059669;font-weight:700;">, within your target</span>' if within else '')}
                  </div>
                </div>
                <div style="text-align:right;min-width:140px;">
                  <div style="font-size:18px;color:#111827;font-weight:800;">£{price_label}</div>
                  <div style="font-size:12px;color:#6b7280;">per passenger</div>
                  <div style="margin-top:6px;">
                    <a href="{view_link}" style="font-size:13px;color:#2563eb;text-decoration:underline;font-weight:700;">
                      View flight
                    </a>
                  </div>
                </div>
              </div>
            </td>
          </tr>
        """

    best_chip = ""
    if best_price_overall is not None:
        best_chip = f"""
        <span style="display:inline-block;padding:8px 12px;border-radius:999px;border:1px solid #e6e8ee;background:#f9fafb;font-size:13px;margin-bottom:8px;">
          Best £{int(best_price_overall)} per passenger
        </span>
        """

    combos_chip = f"""
    <span style="display:inline-block;padding:8px 12px;border-radius:999px;border:1px solid #e6e8ee;background:#f9fafb;font-size:13px;margin-right:8px;margin-bottom:8px;">
      {int(analysed_combinations)} date combinations analysed
    </span>
    """

    html = f"""
    <html>
      <body style="margin:0;padding:0;background:#f6f7f9;font-family:Arial,Helvetica,sans-serif;">
        <div style="max-width:680px;margin:0 auto;padding:24px;">
          <div style="background:#ffffff;border:1px solid #e6e8ee;border-radius:14px;padding:26px;">
            <div style="font-size:14px;color:#6b7280;margin-bottom:10px;">FlyyvFlex Smart Search Alert</div>

            <div style="font-size:28px;line-height:1.2;color:#111827;font-weight:800;margin:0 0 10px 0;">
              Top deals for {origin} → {destination}
            </div>

            <div style="font-size:15px;line-height:1.6;color:#111827;margin:0 0 14px 0;">
              Based on a scan of your <strong>{start_label} to {end_label}</strong> window
              {f" for <strong>{nights_text}-night</strong> trips" if nights_text else ""}.
            </div>

            <div style="font-size:13px;color:#6b7280;margin:0 0 12px 0;">
              Passengers: <strong>{_passengers_label(passengers)}</strong><br>
              Prices shown are per passenger
            </div>

            <div style="margin:0 0 16px 0;">
              <span style="display:inline-block;padding:8px 12px;border-radius:999px;border:1px solid #d1fae5;background:#ecfdf5;font-size:13px;font-weight:800;margin-right:8px;margin-bottom:8px;">
                {str(cabin).upper()}
              </span>
              <span style="display:inline-block;padding:8px 12px;border-radius:999px;border:1px solid #e6e8ee;background:#f9fafb;font-size:13px;margin-right:8px;margin-bottom:8px;">
                {_passengers_label(passengers)}
              </span>
              {combos_chip}
              {best_chip}
            </div>

            <div style="border-top:1px solid #eef0f5;margin:14px 0;"></div>

            <div style="font-size:18px;color:#111827;font-weight:800;margin:0 0 12px 0;">
              Top 5 cheapest date combinations
            </div>

            <table role="presentation" width="100%" cellpadding="0" cellspacing="0" style="border-collapse:separate;">
              {rows_html if rows_html else '<tr><td style="color:#6b7280;font-size:14px;">No flights found in this scan.</td></tr>'}
            </table>

            <div style="margin:18px 0 0 0;">
              <a href="{open_full_results_url}"
                 style="display:inline-block;background:#111827;color:#ffffff;text-decoration:none;padding:12px 16px;border-radius:10px;font-weight:800;font-size:15px;">
                Open full results
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

    msg.add_alternative(html, subtype="html")

    with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
        server.starttls()
        server.login(SMTP_USERNAME, SMTP_PASSWORD)
        server.send_message(msg)

# =====================================================================
# SECTION END: SMART ALERT SUMMARY EMAIL
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

    # Use the same deep link builder so passengers and alertId are always present
    results_url = build_alert_search_link(alert)

    alert_id = _get_attr(alert, "id", None)

    subject = f"{email_type_label}: {origin} \u2192 {destination} | {dep_start_label} to {dep_end_label} | {trip_length_label}"

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = f"FLYYV <{ALERT_FROM_EMAIL}>"
    msg["To"] = to_email

    text_body = (
        f"{email_type_label}\n\n"
        "Your alert is active.\n\n"
        f"Route: {origin} \u2192 {destination}\n"
        f"Cabin: {cabin}\n"
        f"Passengers: {passengers}\n"
        f"{_price_basis_line(passengers)}\n"
        f"Departure window: {dep_window_label}\n"
        f"Trip length: {trip_length_label}\n"
        + (f"Combinations in your window: {theoretical_combinations}\n" if theoretical_combinations else "")
        + (f"Alert ID: {alert_id}\n" if alert_id else "")
        + "\nView results:\n"
        f"{results_url}\n"
    )
    msg.set_content(text_body)

    html = f"""
    <html>
      <body style="margin:0;padding:0;background:#f6f7f9;font-family:Arial,Helvetica,sans-serif;">
        <div style="max-width:680px;margin:0 auto;padding:24px;">
          <div style="background:#ffffff;border:1px solid #e6e8ee;border-radius:14px;padding:26px;">
            <div style="font-size:14px;color:#6b7280;margin-bottom:10px;">{email_type_label}</div>

            <div style="font-size:28px;line-height:1.2;color:#111827;font-weight:700;margin:0 0 12px 0;">
              Your alert is active
            </div>

            <div style="font-size:16px;line-height:1.5;color:#111827;margin:0 0 12px 0;">
              We are watching <strong>{origin} \u2192 {destination}</strong> and will email you when prices match your alert conditions.
            </div>

            <div style="font-size:13px;color:#6b7280;margin:0 0 16px 0;">
              Passengers: <strong>{_passengers_label(passengers)}</strong><br>
              {_price_basis_line(passengers)}
            </div>

            <div style="margin:0 0 18px 0;">
              <span style="display:inline-block;padding:8px 12px;border-radius:999px;border:1px solid #e6e8ee;background:#f9fafb;font-size:13px;margin-right:8px;margin-bottom:8px;">
                {origin} \u2192 {destination}
              </span>
              <span style="display:inline-block;padding:8px 12px;border-radius:999px;border:1px solid #d1fae5;background:#ecfdf5;font-size:13px;font-weight:700;margin-right:8px;margin-bottom:8px;">
                {str(cabin).upper()}
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
                 style="display:inline-block;background:#111827;color:#ffffff;text-decoration:none;padding:12px 16px;border-radius:10px;font-weight:700;font-size:15px;">
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
