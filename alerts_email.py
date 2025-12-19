import os
import smtplib
from email.message import EmailMessage
from datetime import datetime, date, timedelta
from typing import List, Dict, Tuple, Any, Optional, Union

from fastapi import HTTPException
from urllib.parse import urlencode


# ============================================================
# SECTION START: SMTP CONFIG AND CONSTANTS
# ============================================================

SMTP_HOST = os.environ.get("SMTP_HOST", "smtp.smtp2go.com")
SMTP_PORT = int(os.environ.get("SMTP_PORT", "587"))
SMTP_USERNAME = os.environ.get("SMTP_USERNAME")
SMTP_PASSWORD = os.environ.get("SMTP_PASSWORD")
ALERT_FROM_EMAIL = os.environ.get("ALERT_FROM_EMAIL", "alert@flyyv.com")

FRONTEND_BASE_URL = os.environ.get("FRONTEND_BASE_URL", "https://flyyv.com")

# ============================================================
# SECTION END: SMTP CONFIG AND CONSTANTS
# ============================================================


# ============================================================
# SECTION START: LOW LEVEL SMTP SENDER
# ============================================================

def _smtp_ready() -> bool:
    return bool(SMTP_USERNAME and SMTP_PASSWORD and ALERT_FROM_EMAIL)

def _send_email_message(msg: EmailMessage) -> None:
    if not _smtp_ready():
        raise HTTPException(
            status_code=500,
            detail="SMTP settings are not fully configured on the server",
        )

    with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
        server.starttls()
        server.login(SMTP_USERNAME, SMTP_PASSWORD)
        server.send_message(msg)

def send_single_email(to_email: str, subject: str, body: str) -> None:
    """
    Plain text helper for simple one-off emails.
    """
    if not _smtp_ready():
        raise HTTPException(
            status_code=500,
            detail="SMTP settings are not fully configured on the server",
        )

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = f"FLYYV <{ALERT_FROM_EMAIL}>"
    msg["To"] = to_email
    msg.set_content(body)

    _send_email_message(msg)

# ============================================================
# SECTION END: LOW LEVEL SMTP SENDER
# ============================================================


# ============================================================
# SECTION START: ALERT FIELD HELPERS
# ============================================================

def _safe_int(val: Any, default: Optional[int] = None) -> Optional[int]:
    try:
        if val is None:
            return default
        return int(val)
    except Exception:
        return default

def _get_passengers(obj: Any, fallback: int = 1) -> int:
    """
    Alerts and params both may contain passengers.
    We always default to 1 if missing.
    """
    p = getattr(obj, "passengers", None)
    p_int = _safe_int(p, None)
    if p_int is None or p_int < 1:
        return fallback
    return p_int

def _plural(n: int, singular: str, plural: str) -> str:
    return singular if n == 1 else plural

def _format_passengers_label(passengers: int) -> str:
    return f"{passengers} {_plural(passengers, 'passenger', 'passengers')}"

def _prices_explainer(passengers: int) -> str:
    """
    Duffel total_amount represents the total for the offer, based on the passengers sent in the request.
    """
    if passengers <= 1:
        return "Prices shown are the total for all passengers."
    return "Prices shown are the total for all passengers."

def _derive_nights(alert: Any) -> Optional[int]:
    """
    Prefer an explicit nights field, else derive from return_start - departure_start if present.
    """
    n = _safe_int(getattr(alert, "nights", None), None)
    if n is not None and n >= 1:
        return n

    dep_start = getattr(alert, "departure_start", None)
    ret_start = getattr(alert, "return_start", None)
    if dep_start and ret_start:
        try:
            nights = int((ret_start - dep_start).days)
            return max(1, nights)
        except Exception:
            return None

    return None

def _estimate_fixed_night_combinations(
    departure_start: Optional[date],
    departure_end: Optional[date],
    return_end: Optional[date],
    nights: Optional[int],
) -> Optional[int]:
    """
    For fixed-night flex alerts, valid departures are those where dep + nights <= return_end (if return_end exists).
    Example: dep window 01 Jan..15 Jan, nights=7, return_end=15 Jan -> valid deps 01..08 = 8 combos.
    """
    if not departure_start or not departure_end:
        return None

    if nights is None or nights < 1:
        return None

    latest_valid_dep = departure_end
    if return_end:
        try:
            latest_by_return = return_end - timedelta(days=nights)
            if latest_by_return < latest_valid_dep:
                latest_valid_dep = latest_by_return
        except Exception:
            pass

    if latest_valid_dep < departure_start:
        return 0

    try:
        return (latest_valid_dep - departure_start).days + 1
    except Exception:
        return None

def _estimate_combinations(alert: Any, is_flex: bool) -> Optional[int]:
    """
    We keep this simple and accurate for v1:
    - If flex and we can derive fixed nights and return_end, compute accurate fixed-night combos.
    - Else fall back to (departure_end - departure_start + 1) when that is the only safe option.
    """
    if not is_flex:
        return None

    dep_start = getattr(alert, "departure_start", None)
    dep_end = getattr(alert, "departure_end", None)
    ret_end = getattr(alert, "return_end", None)
    nights = _derive_nights(alert)

    combos = _estimate_fixed_night_combinations(dep_start, dep_end, ret_end, nights)
    if combos is not None:
        return combos

    if dep_start and dep_end:
        try:
            approx = (dep_end - dep_start).days + 1
            return approx if approx >= 1 else None
        except Exception:
            return None

    return None

# ============================================================
# SECTION END: ALERT FIELD HELPERS
# ============================================================


# ============================================================
# SECTION START: DEEP LINK BUILDERS
# ============================================================

def _build_searchflyyv_url(qp: Dict[str, Any]) -> str:
    base = FRONTEND_BASE_URL.rstrip("/")
    qp_clean = {k: v for k, v in qp.items() if v is not None and v != ""}
    return f"{base}/SearchFlyyv?{urlencode(qp_clean)}"

def build_flyyv_link_for_date_pair(
    alert: Any,
    departure_iso: str,
    return_iso: str,
    passengers: Optional[int] = None,
) -> str:
    """
    Deep link to a SINGLE date pair result, always searchMode=single.
    """
    pax = passengers if passengers is not None else _get_passengers(alert, fallback=1)

    qp = {
        "origin": getattr(alert, "origin", ""),
        "destination": getattr(alert, "destination", ""),
        "cabin": getattr(alert, "cabin", "BUSINESS"),
        "passengers": str(pax),
        "searchMode": "single",
        "departureStart": departure_iso,
        "departureEnd": departure_iso,
        "returnStart": return_iso,
        "returnEnd": return_iso,
        "autoSearch": "1",
        "alertId": str(getattr(alert, "id", "")) if getattr(alert, "id", None) else None,
    }
    return _build_searchflyyv_url(qp)

def build_alert_search_link(alert: Any) -> str:
    """
    Deep link recreating the ORIGINAL alert context for the full results view.
    Always uses /SearchFlyyv and autoSearch=1.
    """
    mode = (getattr(alert, "mode", None) or "").strip().lower()
    search_mode = (getattr(alert, "search_mode", None) or "").strip().lower()
    is_flex = (mode == "smart") or (search_mode == "flexible")

    pax = _get_passengers(alert, fallback=1)

    qp: Dict[str, Any] = {
        "origin": getattr(alert, "origin", ""),
        "destination": getattr(alert, "destination", ""),
        "cabin": getattr(alert, "cabin", "BUSINESS"),
        "passengers": str(pax),
        "searchMode": "flexible" if is_flex else "single",
        "departureStart": getattr(alert, "departure_start", None).isoformat() if getattr(alert, "departure_start", None) else None,
        "departureEnd": getattr(alert, "departure_end", None).isoformat() if getattr(alert, "departure_end", None) else None,
        "returnStart": getattr(alert, "return_start", None).isoformat() if getattr(alert, "return_start", None) else None,
        "returnEnd": getattr(alert, "return_end", None).isoformat() if getattr(alert, "return_end", None) else None,
        "nights": str(_derive_nights(alert)) if _derive_nights(alert) else None,
        "alertId": str(getattr(alert, "id", "")) if getattr(alert, "id", None) else None,
        "autoSearch": "1",
    }

    return _build_searchflyyv_url(qp)

# ============================================================
# SECTION END: DEEP LINK BUILDERS
# ============================================================


# ============================================================
# SECTION START: ONE OFF ALERT EMAIL (SINGLE DATE PAIR)
# ============================================================

def send_alert_email_for_alert(alert, cheapest, params) -> None:
    """
    One off alert email:
    single date pair, simple format.
    """
    if not _smtp_ready():
        raise HTTPException(
            status_code=500,
            detail="SMTP settings are not fully configured on the server",
        )

    to_email = getattr(alert, "user_email", None)
    if not to_email:
        raise HTTPException(status_code=500, detail="Alert has no user_email")

    passengers = _get_passengers(params, fallback=_get_passengers(alert, fallback=1))

    subject = (
        f"Flyyv Alert: {alert.origin} \u2192 {alert.destination} "
        f"from £{int(cheapest.price)}"
    )

    dep_dt = datetime.fromisoformat(cheapest.departureDate)
    ret_dt = datetime.fromisoformat(cheapest.returnDate)
    dep_label = dep_dt.strftime("%d %b %Y")
    ret_label = ret_dt.strftime("%d %b %Y")

    view_url = build_flyyv_link_for_date_pair(
        alert=alert,
        departure_iso=cheapest.departureDate,
        return_iso=cheapest.returnDate,
        passengers=passengers,
    )

    full_url = build_alert_search_link(alert)

    lines: List[str] = []
    lines.append(f"Route: {alert.origin} \u2192 {alert.destination}, {alert.cabin.title()} class")
    lines.append(f"Passengers: {passengers}")
    lines.append(_prices_explainer(passengers))
    lines.append(f"Dates: {dep_label} to {ret_label}")
    lines.append("")
    lines.append(
        f"Best price found: £{int(cheapest.price)} "
        f"with {cheapest.airline} ({cheapest.airlineCode or ''})"
    )
    lines.append("")
    lines.append("View this flight:")
    lines.append(view_url)
    lines.append("")
    lines.append("Open full results:")
    lines.append(full_url)
    lines.append("")
    lines.append("You are receiving this because you created a Flyyv price alert.")
    lines.append("To stop these alerts, delete the alert in your Flyyv profile.")

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = f"FLYYV <{ALERT_FROM_EMAIL}>"
    msg["To"] = to_email
    msg.set_content("\n".join(lines))

    _send_email_message(msg)

# ============================================================
# SECTION END: ONE OFF ALERT EMAIL (SINGLE DATE PAIR)
# ============================================================


# ============================================================
# SECTION START: SMART ALERT SUMMARY EMAIL (FLYYV FLEX)
# ============================================================

def send_smart_alert_email(alert, options: List, params) -> None:
    """
    FlyyvFlex results email:
    scans multiple date pairs in a flexible window and produces a summary message.

    Requirements:
    - Users care about Best price
    - Show top 5 cheapest date pairs
    - Per-row CTA drills into single date pair results
    - Full results CTA recreates original window
    """
    if not _smtp_ready():
        raise HTTPException(
            status_code=500,
            detail="SMTP settings are not fully configured on the server",
        )

    to_email = getattr(alert, "user_email", None)
    if not to_email:
        raise HTTPException(status_code=500, detail="Alert has no user_email")

    passengers = _get_passengers(params, fallback=_get_passengers(alert, fallback=1))

    threshold = getattr(alert, "max_price", None)
    origin = getattr(alert, "origin", "")
    destination = getattr(alert, "destination", "")

    # Group by date pair
    grouped: Dict[Tuple[str, str], List] = {}
    for opt in options:
        key = (getattr(opt, "departureDate", None), getattr(opt, "returnDate", None))
        if key[0] and key[1]:
            grouped.setdefault(key, []).append(opt)

    any_under = False
    pairs_summary: List[Dict[str, Any]] = []

    for (dep_iso, ret_iso), flights in grouped.items():
        prices = [o.price for o in flights if getattr(o, "price", None) is not None]
        if not prices:
            continue

        min_price = min(prices)
        cheapest = min(flights, key=lambda o: o.price)

        flights_under: List = []
        if threshold is not None:
            try:
                flights_under = [o for o in flights if o.price <= float(threshold)]
                if flights_under:
                    any_under = True
            except Exception:
                flights_under = []

        flyyv_link = build_flyyv_link_for_date_pair(
            alert=alert,
            departure_iso=dep_iso,
            return_iso=ret_iso,
            passengers=passengers,
        )

        pairs_summary.append(
            {
                "departureDate": dep_iso,
                "returnDate": ret_iso,
                "totalFlights": len(flights),
                "cheapestPrice": cheapest.price,
                "cheapestAirline": getattr(cheapest, "airline", None),
                "flightsUnderThresholdCount": len(flights_under),
                "flyyvLink": flyyv_link,
                "minPrice": min_price,
            }
        )

    start_label = params.earliestDeparture.strftime("%d %b %Y")
    end_label = params.latestDeparture.strftime("%d %b %Y")

    nights_text = None
    try:
        nights_val = _derive_nights(alert)
        if nights_val:
            nights_text = str(nights_val)
    except Exception:
        nights_text = None

    combinations_checked = len(pairs_summary)

    best_price_overall = None
    if pairs_summary:
        try:
            best_price_overall = int(min(pairs_summary, key=lambda x: x["cheapestPrice"])["cheapestPrice"])
        except Exception:
            best_price_overall = None

    if best_price_overall is not None:
        subject = f"FlyyvFlex Alert: {origin} \u2192 {destination} from £{best_price_overall}"
    elif threshold is not None and any_under:
        subject = f"FlyyvFlex Alert: {origin} \u2192 {destination} fares under £{int(threshold)}"
    else:
        subject = f"FlyyvFlex Alert: {origin} \u2192 {destination} update"

    top_pairs = [
        p for p in pairs_summary
        if p.get("totalFlights", 0) > 0 and p.get("cheapestPrice") is not None
    ]

    MAX_RESULTS = 5
    top_pairs_sorted = sorted(top_pairs, key=lambda x: x["cheapestPrice"])[:MAX_RESULTS]

    open_full_results_url = build_alert_search_link(alert)

    # Plain text
    lines: List[str] = []
    lines.append("FlyyvFlex Smart Search Alert")
    lines.append(f"Route: {origin} \u2192 {destination}, {getattr(alert, 'cabin', 'BUSINESS').title()} class")
    lines.append(f"Passengers: {passengers}")
    lines.append(_prices_explainer(passengers))
    if nights_text:
        lines.append(f"Trip length: {nights_text} nights")
    lines.append(f"Date window: {start_label} to {end_label}")
    lines.append(f"Combinations checked: {combinations_checked}")
    if best_price_overall is not None:
        lines.append(f"Best price found: £{best_price_overall}")
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
            lines.append(f"£{price_label} | {dep_label} to {ret_label} | {airline_label}")
            lines.append(f"View flight: {p.get('flyyvLink')}")
            lines.append("")

    lines.append("Open full results:")
    lines.append(open_full_results_url)

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = f"FLYYV <{ALERT_FROM_EMAIL}>"
    msg["To"] = to_email
    msg.set_content("\n".join(lines))

    # HTML rows
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
                    {origin} \u2192 {destination}
                  </div>
                  <div style="font-size:13px;color:#6b7280;margin-bottom:6px;">
                    {dep_label} to {ret_label}
                  </div>
                  <div style="font-size:13px;color:#111827;">
                    Cheapest option: <strong>{airline_label}</strong>
                    {('<span style="color:#059669;font-weight:700;">, within your limit</span>' if within else '')}
                  </div>
                </div>
                <div style="text-align:right;min-width:120px;">
                  <div style="font-size:18px;color:#111827;font-weight:800;">£{price_label}</div>
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
          Best £{int(best_price_overall)}
        </span>
        """

    html = f"""
    <html>
      <body style="margin:0;padding:0;background:#f6f7f9;font-family:Arial,Helvetica,sans-serif;">
        <div style="max-width:680px;margin:0 auto;padding:24px;">
          <div style="background:#ffffff;border:1px solid #e6e8ee;border-radius:14px;padding:26px;">
            <div style="font-size:14px;color:#6b7280;margin-bottom:10px;">FlyyvFlex Smart Search Alert</div>

            <div style="font-size:28px;line-height:1.2;color:#111827;font-weight:800;margin:0 0 10px 0;">
              Top deals for {origin} \u2192 {destination}
            </div>

            <div style="font-size:15px;line-height:1.6;color:#111827;margin:0 0 10px 0;">
              Based on a full scan of your <strong>{start_label} to {end_label}</strong> window
              {f" for <strong>{nights_text}-night</strong> trips" if nights_text else ""}.
            </div>

            <div style="font-size:13px;line-height:1.5;color:#6b7280;margin:0 0 14px 0;">
              {_prices_explainer(passengers)}
            </div>

            <div style="margin:0 0 16px 0;">
              <span style="display:inline-block;padding:8px 12px;border-radius:999px;border:1px solid #d1fae5;background:#ecfdf5;font-size:13px;font-weight:800;margin-right:8px;margin-bottom:8px;">
                {getattr(alert, "cabin", "BUSINESS")}
              </span>
              <span style="display:inline-block;padding:8px 12px;border-radius:999px;border:1px solid #e6e8ee;background:#f9fafb;font-size:13px;margin-right:8px;margin-bottom:8px;">
                {_format_passengers_label(passengers)}
              </span>
              <span style="display:inline-block;padding:8px 12px;border-radius:999px;border:1px solid #e6e8ee;background:#f9fafb;font-size:13px;margin-right:8px;margin-bottom:8px;">
                {combinations_checked} combinations
              </span>
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
    _send_email_message(msg)

# ============================================================
# SECTION END: SMART ALERT SUMMARY EMAIL (FLYYV FLEX)
# ============================================================


# ============================================================
# SECTION START: ALERT CONFIRMATION EMAIL
# ============================================================

def send_alert_confirmation_email(alert) -> None:
    """
    Sent immediately when a user creates an alert.
    Confirms the alert is active.
    """
    if not _smtp_ready():
        return

    to_email = getattr(alert, "user_email", None)
    if not to_email:
        return

    origin = getattr(alert, "origin", "")
    destination = getattr(alert, "destination", "")
    cabin = getattr(alert, "cabin", "BUSINESS")

    departure_start = getattr(alert, "departure_start", None)
    departure_end = getattr(alert, "departure_end", None)
    return_start = getattr(alert, "return_start", None)
    return_end = getattr(alert, "return_end", None)

    search_mode = (getattr(alert, "search_mode", None) or "").strip().lower()
    mode = (getattr(alert, "mode", None) or "").strip().lower()
    is_flex = (mode == "smart") or (search_mode == "flexible")

    passengers = _get_passengers(alert, fallback=1)

    email_type_label = "FlyyvFlex Smart Search Alert" if is_flex else "Flyyv Alert"
    pill_type_label = "Smart price watch" if is_flex else "Price alert"

    dep_start_label = departure_start.strftime("%d %b %Y") if departure_start else "Not set"
    dep_end_label = departure_end.strftime("%d %b %Y") if departure_end else "Not set"
    dep_window_label = f"{dep_start_label} to {dep_end_label}"

    trip_length_label = "Flexible"
    try:
        nights_val = _derive_nights(alert)
        if nights_val:
            trip_length_label = f"{nights_val} nights"
        elif not is_flex:
            trip_length_label = "Not set"
    except Exception:
        trip_length_label = "Flexible"

    combinations_checked = _estimate_combinations(alert, is_flex=is_flex)

    results_url = build_alert_search_link(alert)
    alert_id = getattr(alert, "id", None)

    subject = f"{email_type_label}: {origin} \u2192 {destination} | {dep_start_label} to {dep_end_label} | {trip_length_label}"

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = f"FLYYV <{ALERT_FROM_EMAIL}>"
    msg["To"] = to_email

    text_lines: List[str] = []
    text_lines.append(email_type_label)
    text_lines.append("")
    text_lines.append("Your alert is active.")
    text_lines.append("")
    text_lines.append(f"Route: {origin} \u2192 {destination}")
    text_lines.append(f"Cabin: {cabin}")
    text_lines.append(f"Passengers: {passengers}")
    text_lines.append(_prices_explainer(passengers))
    text_lines.append(f"Departure window: {dep_window_label}")
    text_lines.append(f"Trip length: {trip_length_label}")
    if combinations_checked is not None:
        text_lines.append(f"Combinations: {combinations_checked}")
    if alert_id:
        text_lines.append(f"Alert ID: {alert_id}")
    text_lines.append("")
    text_lines.append("View results:")
    text_lines.append(results_url)

    msg.set_content("\n".join(text_lines))

    combinations_chip_html = ""
    if combinations_checked is not None:
        combinations_chip_html = f"""
        <span style="display:inline-block;padding:8px 12px;border-radius:999px;border:1px solid #e6e8ee;background:#f9fafb;font-size:13px;margin-bottom:8px;">
          {combinations_checked} combinations
        </span>
        """

    html = f"""
    <html>
      <body style="margin:0;padding:0;background:#f6f7f9;font-family:Arial,Helvetica,sans-serif;">
        <div style="max-width:680px;margin:0 auto;padding:24px;">
          <div style="background:#ffffff;border:1px solid #e6e8ee;border-radius:14px;padding:26px;">
            <div style="font-size:14px;color:#6b7280;margin-bottom:10px;">{email_type_label}</div>

            <div style="font-size:28px;line-height:1.2;color:#111827;font-weight:700;margin:0 0 12px 0;">
              Your alert is active
            </div>

            <div style="font-size:16px;line-height:1.5;color:#111827;margin:0 0 10px 0;">
              We are watching <strong>{origin} \u2192 {destination}</strong> and will email you when prices match your alert conditions.
            </div>

            <div style="font-size:13px;line-height:1.5;color:#6b7280;margin:0 0 16px 0;">
              {_prices_explainer(passengers)}
            </div>

            <div style="margin:0 0 18px 0;">
              <span style="display:inline-block;padding:8px 12px;border-radius:999px;border:1px solid #e6e8ee;background:#f9fafb;font-size:13px;margin-right:8px;margin-bottom:8px;">
                {origin} \u2192 {destination}
              </span>
              <span style="display:inline-block;padding:8px 12px;border-radius:999px;border:1px solid #d1fae5;background:#ecfdf5;font-size:13px;font-weight:700;margin-right:8px;margin-bottom:8px;">
                {cabin}
              </span>
              <span style="display:inline-block;padding:8px 12px;border-radius:999px;border:1px solid #e6e8ee;background:#f9fafb;font-size:13px;margin-right:8px;margin-bottom:8px;">
                {_format_passengers_label(passengers)}
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
              {combinations_chip_html}
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
        _send_email_message(msg)
    except Exception:
        pass

# ============================================================
# SECTION END: ALERT CONFIRMATION EMAIL
# ============================================================


# ============================================================
# SECTION START: EARLY ACCESS WELCOME EMAIL
# ============================================================

def send_early_access_welcome_email(to_email: str):
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

    send_single_email(
        to_email=to_email,
        subject=subject,
        body=body,
    )

# ============================================================
# SECTION END: EARLY ACCESS WELCOME EMAIL
# ============================================================
