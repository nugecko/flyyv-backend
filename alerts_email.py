import os
import smtplib
from email.message import EmailMessage
from datetime import datetime, date, timedelta
from typing import List, Dict, Tuple, Any, Optional, Union

from fastapi import HTTPException
from urllib.parse import urlencode


# =======================================
# SECTION START: SMTP CONFIG AND CONSTANTS
# =======================================

SMTP_HOST = os.environ.get("SMTP_HOST", "smtp.smtp2go.com")
SMTP_PORT = int(os.environ.get("SMTP_PORT", "587"))
SMTP_USERNAME = os.environ.get("SMTP_USERNAME")
SMTP_PASSWORD = os.environ.get("SMTP_PASSWORD")
ALERT_FROM_EMAIL = os.environ.get("ALERT_FROM_EMAIL", "alert@flyyv.com")

FRONTEND_BASE_URL = os.environ.get("FRONTEND_BASE_URL", "https://flyyv.com")

# =======================================
# SECTION END: SMTP CONFIG AND CONSTANTS
# =======================================


# =======================================
# SECTION START: EMAIL SENDING HELPERS
# =======================================

def _smtp_ready() -> bool:
    return bool(SMTP_USERNAME and SMTP_PASSWORD and ALERT_FROM_EMAIL)

def send_email(to_email: str, subject: str, text_body: str, html_body: Optional[str] = None) -> None:
    if not _smtp_ready():
        raise HTTPException(
            status_code=500,
            detail="SMTP settings are not fully configured on the server",
        )

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = f"FLYYV <{ALERT_FROM_EMAIL}>"
    msg["To"] = to_email
    msg.set_content(text_body)

    if html_body:
        msg.add_alternative(html_body, subtype="html")

    with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
        server.starttls()
        server.login(SMTP_USERNAME, SMTP_PASSWORD)
        server.send_message(msg)

# =======================================
# SECTION END: EMAIL SENDING HELPERS
# =======================================


# =======================================
# SECTION START: GENERIC FIELD HELPERS
# =======================================

def get_int(obj: Any, field: str, default: int) -> int:
    try:
        val = getattr(obj, field, None)
        if val is None:
            return default
        return int(val)
    except Exception:
        return default

def get_str(obj: Any, field: str, default: str = "") -> str:
    try:
        val = getattr(obj, field, None)
        return default if val is None else str(val)
    except Exception:
        return default

def safe_date_label(d: Optional[date]) -> str:
    if not d:
        return "Not set"
    return d.strftime("%d %b %Y")

def compute_trip_nights(alert: Any) -> Optional[int]:
    """
    Best effort:
    - If return_start and departure_start exist, use the difference
    - Else if alert.nights exists, use it
    """
    try:
        dep = getattr(alert, "departure_start", None)
        ret = getattr(alert, "return_start", None)
        if dep and ret:
            return max(1, (ret - dep).days)
    except Exception:
        pass

    try:
        nights = getattr(alert, "nights", None)
        if nights is not None:
            return max(1, int(nights))
    except Exception:
        pass

    return None

def compute_flex_combinations(alert: Any) -> Optional[int]:
    """
    For fixed-night flexible alerts, combinations should represent how many departure dates are valid.
    We assume the scan is for dep in [departure_start..departure_end] with a fixed nights length.
    If return_end exists, we only count departure dates where dep + nights <= return_end.

    If nights is not known, fall back to a simple inclusive window size.
    """
    dep_start = getattr(alert, "departure_start", None)
    dep_end = getattr(alert, "departure_end", None)
    if not dep_start or not dep_end:
        return None

    if dep_end < dep_start:
        return None

    nights = compute_trip_nights(alert)
    ret_end = getattr(alert, "return_end", None)

    total_days = (dep_end - dep_start).days + 1
    if total_days < 1:
        return None

    if not nights:
        return total_days

    if not ret_end:
        # No explicit return_end, count all departure days in the dep window
        return total_days

    # Count only departure dates that keep the return inside the return window
    valid = 0
    cur = dep_start
    while cur <= dep_end:
        ret = cur + timedelta(days=nights)
        if ret <= ret_end:
            valid += 1
        cur += timedelta(days=1)

    return valid if valid > 0 else None

def format_price(total_price_gbp: int, passengers: int) -> Tuple[str, Optional[str]]:
    """
    Duffel total_amount reflects the total for the requested passengers.
    We show total, and optionally per-person if passengers > 1.
    """
    total_label = f"£{int(total_price_gbp)}"
    if passengers and passengers > 1:
        try:
            per_person = int(round(total_price_gbp / float(passengers)))
            return total_label, f"~£{per_person} pp"
        except Exception:
            return total_label, None
    return total_label, None

# =======================================
# SECTION END: GENERIC FIELD HELPERS
# =======================================


# =======================================
# SECTION START: HELPER LINK BUILDERS
# =======================================

def build_flyyv_link(obj: Any, departure: str, return_date: str) -> str:
    """
    Builds a deep link to Flyyv search results for a specific date pair.
    Always uses /SearchFlyyv and autoSearch=1.
    Drilldowns must use searchMode=single.
    Includes passengers when available to prevent frontend defaulting to 1 pax.
    """
    base = FRONTEND_BASE_URL.rstrip("/")

    origin = get_str(obj, "origin")
    destination = get_str(obj, "destination")
    cabin = get_str(obj, "cabin", "BUSINESS")
    passengers = get_int(obj, "passengers", 1)

    qp: Dict[str, Any] = {
        "origin": origin,
        "destination": destination,
        "cabin": cabin,
        "passengers": str(passengers),
        "searchMode": "single",
        "departureStart": departure,
        "departureEnd": departure,
        "returnStart": return_date,
        "returnEnd": return_date,
        "autoSearch": "1",
    }

    alert_id = getattr(obj, "id", None) or getattr(obj, "alertId", None) or getattr(obj, "alert_id", None)
    if alert_id is not None:
        qp["alertId"] = str(alert_id)

    return f"{base}/SearchFlyyv?{urlencode(qp)}"


def build_alert_search_link(alert: Any) -> str:
    """
    Builds the full results deep link for the original alert window.
    Includes passengers so the frontend runs the correct pax.
    """
    base = FRONTEND_BASE_URL.rstrip("/")

    origin = get_str(alert, "origin")
    destination = get_str(alert, "destination")
    cabin = get_str(alert, "cabin", "BUSINESS")
    passengers = get_int(alert, "passengers", 1)

    mode = (get_str(alert, "mode") or "").strip().lower()
    search_mode = (get_str(alert, "search_mode") or "").strip().lower()
    is_flex = (mode == "smart") or (search_mode == "flexible")

    dep_start = getattr(alert, "departure_start", None)
    dep_end = getattr(alert, "departure_end", None)
    ret_start = getattr(alert, "return_start", None)
    ret_end = getattr(alert, "return_end", None)

    qp: Dict[str, Any] = {
        "origin": origin,
        "destination": destination,
        "cabin": cabin,
        "passengers": str(passengers),
        "searchMode": "flexible" if is_flex else "single",
        "departureStart": dep_start.isoformat() if dep_start else None,
        "departureEnd": dep_end.isoformat() if dep_end else None,
        "returnStart": ret_start.isoformat() if ret_start else None,
        "returnEnd": ret_end.isoformat() if ret_end else None,
        "alertId": getattr(alert, "id", None),
        "autoSearch": "1",
    }

    nights = compute_trip_nights(alert)
    if nights:
        qp["nights"] = str(nights)

    qp = {k: v for k, v in qp.items() if v is not None and v != ""}

    return f"{base}/SearchFlyyv?{urlencode(qp)}"

# =======================================
# SECTION END: HELPER LINK BUILDERS
# =======================================


# =======================================
# SECTION START: ONE OFF ALERT EMAIL
# =======================================

def send_alert_email_for_alert(alert: Any, cheapest: Any, params: Any) -> None:
    """
    One off alert email:
    single date pair, simple format.
    """
    to_email = getattr(alert, "user_email", None)
    if not to_email:
        raise HTTPException(status_code=500, detail="Alert has no user_email")

    passengers = get_int(alert, "passengers", 1)

    price_total = int(getattr(cheapest, "price", 0))
    total_label, pp_label = format_price(price_total, passengers)

    subject = f"Flyyv Alert: {alert.origin} \u2192 {alert.destination} from {total_label}"

    dep_dt = datetime.fromisoformat(getattr(cheapest, "departureDate"))
    ret_dt = datetime.fromisoformat(getattr(cheapest, "returnDate"))
    dep_label = dep_dt.strftime("%d %b %Y")
    ret_label = ret_dt.strftime("%d %b %Y")

    lines: List[str] = []
    lines.append(f"Route: {alert.origin} \u2192 {alert.destination}, {get_str(alert, 'cabin', 'BUSINESS').title()} class")
    lines.append(f"Passengers: {passengers}")
    lines.append("Price shown is the total for all passengers.")
    if pp_label:
        lines.append(f"Approx per person: {pp_label.replace('~', '')}")
    lines.append(f"Dates: {dep_label} to {ret_label}")
    lines.append("")
    lines.append(f"Best price found: {total_label} with {get_str(cheapest, 'airline')} ({get_str(cheapest, 'airlineCode')})")
    lines.append("")
    lines.append("View results:")
    lines.append(build_flyyv_link(alert, getattr(cheapest, "departureDate"), getattr(cheapest, "returnDate")))
    lines.append("")
    lines.append("You are receiving this because you created a Flyyv price alert.")
    lines.append("To stop these alerts, delete the alert in your Flyyv profile.")

    send_email(to_email=to_email, subject=subject, text_body="\n".join(lines), html_body=None)

# =======================================
# SECTION END: ONE OFF ALERT EMAIL
# =======================================


# =======================================
# SECTION START: SMART ALERT SUMMARY EMAIL
# =======================================

def send_smart_alert_email(alert: Any, options: List[Any], params: Any) -> None:
    """
    FlyyvFlex results email:
    scans multiple date pairs in a flexible window and produces a summary message.

    Requirements:
    - Users care about Best price, not Max
    - Show top 5 cheapest date pairs
    - Per-row CTA drills into single date pair results
    - Full results CTA recreates original window
    """
    to_email = getattr(alert, "user_email", None)
    if not to_email:
        raise HTTPException(status_code=500, detail="Alert has no user_email")

    threshold = getattr(alert, "max_price", None)
    origin = get_str(alert, "origin")
    destination = get_str(alert, "destination")
    cabin = get_str(alert, "cabin", "BUSINESS")
    passengers = get_int(alert, "passengers", 1)

    grouped: Dict[Tuple[str, str], List[Any]] = {}
    for opt in options:
        key = (getattr(opt, "departureDate", None), getattr(opt, "returnDate", None))
        if key[0] and key[1]:
            grouped.setdefault(key, []).append(opt)

    any_under = False
    pairs_summary: List[Dict[str, Any]] = []

    for dep_iso, ret_iso in grouped.keys():
        flights = grouped[(dep_iso, ret_iso)]
        prices = [getattr(o, "price", None) for o in flights if getattr(o, "price", None) is not None]
        if not prices:
            continue

        min_price = min(prices)
        cheapest = min(flights, key=lambda o: getattr(o, "price", 10**18))

        flights_under: List[Any] = []
        if threshold is not None:
            try:
                flights_under = [o for o in flights if float(getattr(o, "price", 0)) <= float(threshold)]
            except Exception:
                flights_under = []
            if flights_under:
                any_under = True

        flyyv_link = build_flyyv_link(alert, dep_iso, ret_iso)

        pairs_summary.append(
            {
                "departureDate": dep_iso,
                "returnDate": ret_iso,
                "totalFlights": len(flights),
                "cheapestPrice": float(getattr(cheapest, "price", 0)),
                "cheapestAirline": getattr(cheapest, "airline", None),
                "flyyvLink": flyyv_link,
                "minPrice": float(min_price),
            }
        )

    start_label = params.earliestDeparture.strftime("%d %b %Y")
    end_label = params.latestDeparture.strftime("%d %b %Y")

    nights_val = compute_trip_nights(alert)
    nights_text = str(nights_val) if nights_val else None

    combinations_checked = len(pairs_summary)

    best_price_overall: Optional[int] = None
    if pairs_summary:
        try:
            best_price_overall = int(min(pairs_summary, key=lambda x: x["cheapestPrice"])["cheapestPrice"])
        except Exception:
            best_price_overall = None

    if best_price_overall is not None:
        subject_total, _pp = format_price(best_price_overall, passengers)
        subject = f"FlyyvFlex Alert: {origin} \u2192 {destination} from {subject_total}"
    elif threshold is not None and any_under:
        subject = f"FlyyvFlex Alert: {origin} \u2192 {destination} fares under £{int(threshold)}"
    else:
        subject = f"FlyyvFlex Alert: {origin} \u2192 {destination} update"

    top_pairs = [
        p for p in pairs_summary
        if p.get("totalFlights", 0) > 0 and p.get("cheapestPrice") is not None
    ]

    top_pairs_sorted = sorted(top_pairs, key=lambda x: x["cheapestPrice"])[:5]
    open_full_results_url = build_alert_search_link(alert)

    # Plain text
    lines: List[str] = []
    lines.append("FlyyvFlex Smart Search Alert")
    lines.append(f"Route: {origin} \u2192 {destination}, {cabin.title()} class")
    lines.append(f"Passengers: {passengers}")
    lines.append("Prices shown are the total for all passengers.")
    if nights_text:
        lines.append(f"Trip length: {nights_text} nights")
    lines.append(f"Date window: {start_label} to {end_label}")
    lines.append(f"Combinations checked: {combinations_checked}")
    if best_price_overall is not None:
        best_total, best_pp = format_price(best_price_overall, passengers)
        if best_pp:
            lines.append(f"Best price found: {best_total} ({best_pp})")
        else:
            lines.append(f"Best price found: {best_total}")
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

            price_total = int(p["cheapestPrice"])
            total_label, pp_label = format_price(price_total, passengers)

            airline_label = p.get("cheapestAirline") or "Multiple airlines"

            if pp_label:
                lines.append(f"{total_label} ({pp_label}) | {dep_label} to {ret_label} | {airline_label}")
            else:
                lines.append(f"{total_label} | {dep_label} to {ret_label} | {airline_label}")

            lines.append(f"View flight: {p.get('flyyvLink')}")
            lines.append("")

    lines.append("Open full results:")
    lines.append(open_full_results_url)
    text_body = "\n".join(lines)

    # HTML rows
    rows_html = ""
    for p in top_pairs_sorted:
        dep_dt = datetime.fromisoformat(p["departureDate"])
        ret_dt = datetime.fromisoformat(p["returnDate"])
        dep_label = dep_dt.strftime("%d %b %Y")
        ret_label = ret_dt.strftime("%d %b %Y")

        price_total = int(p["cheapestPrice"])
        total_label, pp_label = format_price(price_total, passengers)

        airline_label = p.get("cheapestAirline") or "Multiple airlines"
        view_link = p.get("flyyvLink") or open_full_results_url

        within = False
        try:
            if threshold is not None and float(price_total) <= float(threshold):
                within = True
        except Exception:
            within = False

        pp_html = f'<div style="font-size:12px;color:#6b7280;margin-top:2px;">{pp_label}</div>' if pp_label else ""

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
                <div style="text-align:right;min-width:140px;">
                  <div style="font-size:18px;color:#111827;font-weight:800;">{total_label}</div>
                  {pp_html}
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
        best_total, best_pp = format_price(best_price_overall, passengers)
        best_chip = f"""
        <span style="display:inline-block;padding:8px 12px;border-radius:999px;border:1px solid #e6e8ee;background:#f9fafb;font-size:13px;margin-bottom:8px;">
          Best {best_total}{f" ({best_pp})" if best_pp else ""}
        </span>
        """

    passengers_chip = f"""
      <span style="display:inline-block;padding:8px 12px;border-radius:999px;border:1px solid #e6e8ee;background:#f9fafb;font-size:13px;margin-right:8px;margin-bottom:8px;">
        {passengers} passenger{'' if passengers == 1 else 's'}
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

            <div style="font-size:15px;line-height:1.6;color:#111827;margin:0 0 6px 0;">
              Based on a full scan of your <strong>{start_label} to {end_label}</strong> window
              {f" for <strong>{nights_text}-night</strong> trips" if nights_text else ""}.
            </div>

            <div style="font-size:13px;color:#6b7280;margin:0 0 14px 0;">
              Prices shown are the total for all passengers.
            </div>

            <div style="margin:0 0 16px 0;">
              <span style="display:inline-block;padding:8px 12px;border-radius:999px;border:1px solid #d1fae5;background:#ecfdf5;font-size:13px;font-weight:800;margin-right:8px;margin-bottom:8px;">
                {cabin}
              </span>
              {passengers_chip}
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

    send_email(to_email=to_email, subject=subject, text_body=text_body, html_body=html)

# =======================================
# SECTION END: SMART ALERT SUMMARY EMAIL
# =======================================


# =======================================
# SECTION START: ALERT CONFIRMATION EMAIL
# =======================================

def send_alert_confirmation_email(alert: Any) -> None:
    """
    Sent immediately when a user creates an alert.
    Confirms the alert is active.
    """
    if not _smtp_ready():
        return

    to_email = getattr(alert, "user_email", None)
    if not to_email:
        return

    origin = get_str(alert, "origin")
    destination = get_str(alert, "destination")
    cabin = get_str(alert, "cabin", "BUSINESS")
    passengers = get_int(alert, "passengers", 1)

    departure_start = getattr(alert, "departure_start", None)
    departure_end = getattr(alert, "departure_end", None)
    return_start = getattr(alert, "return_start", None)
    return_end = getattr(alert, "return_end", None)

    search_mode = (get_str(alert, "search_mode") or "").strip().lower()
    mode = (get_str(alert, "mode") or "").strip().lower()
    is_flex = (mode == "smart") or (search_mode == "flexible")

    email_type_label = "FlyyvFlex Smart Search Alert" if is_flex else "Flyyv Alert"
    pill_type_label = "Smart price watch" if is_flex else "Price alert"

    dep_start_label = safe_date_label(departure_start)
    dep_end_label = safe_date_label(departure_end)
    dep_window_label = f"{dep_start_label} to {dep_end_label}"

    nights = compute_trip_nights(alert)
    trip_length_label = f"{nights} nights" if nights else ("Flexible" if is_flex else "Not set")

    combinations_checked = compute_flex_combinations(alert) if is_flex else None

    base = FRONTEND_BASE_URL.rstrip("/")
    results_qp: Dict[str, Any] = {
        "origin": origin,
        "destination": destination,
        "cabin": cabin,
        "passengers": str(passengers),
        "searchMode": "flexible" if is_flex else "single",
        "departureStart": departure_start.isoformat() if departure_start else "",
        "departureEnd": departure_end.isoformat() if departure_end else "",
        "autoSearch": "1",
        "alertId": getattr(alert, "id", None),
    }

    if return_start:
        results_qp["returnStart"] = return_start.isoformat()
    if return_end:
        results_qp["returnEnd"] = return_end.isoformat()
    if nights:
        results_qp["nights"] = str(nights)

    results_qp = {k: v for k, v in results_qp.items() if v not in (None, "")}
    results_url = f"{base}/SearchFlyyv?{urlencode(results_qp)}"

    alert_id = getattr(alert, "id", None)

    subject = f"{email_type_label}: {origin} \u2192 {destination} | {dep_start_label} to {dep_end_label} | {trip_length_label}"

    # Plain text
    text_lines: List[str] = []
    text_lines.append(email_type_label)
    text_lines.append("")
    text_lines.append("Your alert is active.")
    text_lines.append("")
    text_lines.append(f"Route: {origin} \u2192 {destination}")
    text_lines.append(f"Cabin: {cabin}")
    text_lines.append(f"Passengers: {passengers}")
    text_lines.append("Prices shown are the total for all passengers.")
    text_lines.append(f"Departure window: {dep_window_label}")
    text_lines.append(f"Trip length: {trip_length_label}")
    if combinations_checked:
        text_lines.append(f"Combinations: {combinations_checked}")
    if alert_id:
        text_lines.append(f"Alert ID: {alert_id}")
    text_lines.append("")
    text_lines.append("View results:")
    text_lines.append(results_url)
    text_body = "\n".join(text_lines)

    # HTML
    passengers_label = f"{passengers} passenger" if passengers == 1 else f"{passengers} passengers"

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

            <div style="font-size:13px;color:#6b7280;margin:0 0 16px 0;">
              Prices shown are the total for all passengers.
            </div>

            <div style="margin:0 0 18px 0;">
              <span style="display:inline-block;padding:8px 12px;border-radius:999px;border:1px solid #e6e8ee;background:#f9fafb;font-size:13px;margin-right:8px;margin-bottom:8px;">
                {origin} \u2192 {destination}
              </span>
              <span style="display:inline-block;padding:8px 12px;border-radius:999px;border:1px solid #d1fae5;background:#ecfdf5;font-size:13px;font-weight:700;margin-right:8px;margin-bottom:8px;">
                {cabin}
              </span>
              <span style="display:inline-block;padding:8px 12px;border-radius:999px;border:1px solid #e6e8ee;background:#f9fafb;font-size:13px;margin-right:8px;margin-bottom:8px;">
                {passengers_label}
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
                {combinations_checked} combinations
              </span>
              ''' if combinations_checked else ''}
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

    try:
        send_email(to_email=to_email, subject=subject, text_body=text_body, html_body=html)
    except Exception:
        pass

# =======================================
# SECTION END: ALERT CONFIRMATION EMAIL
# =======================================


# =======================================
# SECTION START: EARLY ACCESS WELCOME EMAIL
# =======================================

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

    send_email(to_email=to_email, subject=subject, text_body=body, html_body=None)

# =======================================
# SECTION END: EARLY ACCESS WELCOME EMAIL
# =======================================
