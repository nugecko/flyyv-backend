import os
import smtplib
from email.message import EmailMessage
from datetime import datetime, timedelta, date
from typing import List, Dict, Tuple, Any, Optional
from urllib.parse import urlencode

from fastapi import HTTPException

# =======================================
# START SECTION: SMTP CONFIG AND CONSTANTS
# =======================================

SMTP_HOST = os.environ.get("SMTP_HOST", "smtp.smtp2go.com")
SMTP_PORT = int(os.environ.get("SMTP_PORT", "587"))
SMTP_USERNAME = os.environ.get("SMTP_USERNAME")
SMTP_PASSWORD = os.environ.get("SMTP_PASSWORD")
ALERT_FROM_EMAIL = os.environ.get("ALERT_FROM_EMAIL", "alert@flyyv.com")

FRONTEND_BASE_URL = os.environ.get("FRONTEND_BASE_URL", "https://flyyv.com")

# =====================================
# END SECTION: SMTP CONFIG AND CONSTANTS
# =====================================


# =======================================
# START SECTION: SMTP SENDER UTILITIES
# =======================================

def _smtp_ready_or_raise() -> None:
    if not (SMTP_USERNAME and SMTP_PASSWORD and ALERT_FROM_EMAIL):
        raise HTTPException(
            status_code=500,
            detail="SMTP settings are not fully configured on the server",
        )


def _send_email_message(msg: EmailMessage) -> None:
    _smtp_ready_or_raise()
    with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
        server.starttls()
        server.login(SMTP_USERNAME, SMTP_PASSWORD)
        server.send_message(msg)


def send_single_email(to_email: str, subject: str, body: str) -> None:
    _smtp_ready_or_raise()
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = f"FLYYV <{ALERT_FROM_EMAIL}>"
    msg["To"] = to_email
    msg.set_content(body)
    _send_email_message(msg)

# =====================================
# END SECTION: SMTP SENDER UTILITIES
# =====================================


# =======================================
# START SECTION: LINK BUILDERS
# =======================================

def build_flyyv_link(params, departure: str, return_date: str) -> str:
    """
    Deep link to a specific date pair result.
    Must use /SearchFlyyv and autoSearch=1.
    Must use searchMode=single for drilldowns.
    """
    base = FRONTEND_BASE_URL.rstrip("/")

    qp = {
        "origin": params.origin,
        "destination": params.destination,
        "cabin": params.cabin,
        "searchMode": "single",
        "departureStart": departure,
        "departureEnd": departure,
        "returnStart": return_date,
        "returnEnd": return_date,
        "autoSearch": "1",
    }

    # Pass through alertId when available
    alert_id = (
        getattr(params, "alertId", None)
        or getattr(params, "alert_id", None)
        or getattr(params, "id", None)
    )
    if alert_id is not None:
        qp["alertId"] = str(alert_id)

    return f"{base}/SearchFlyyv?{urlencode(qp)}"


def build_alert_search_link(alert) -> str:
    """
    Deep link to the original alert window.
    Must recreate the window context.
    """
    base = FRONTEND_BASE_URL.rstrip("/")

    is_flex = (
        getattr(alert, "mode", None) == "smart"
        or getattr(alert, "search_mode", None) == "flexible"
    )

    qp = {
        "origin": getattr(alert, "origin", ""),
        "destination": getattr(alert, "destination", ""),
        "cabin": getattr(alert, "cabin", "BUSINESS"),
        "searchMode": "flexible" if is_flex else "single",
        "departureStart": alert.departure_start.isoformat() if getattr(alert, "departure_start", None) else None,
        "departureEnd": alert.departure_end.isoformat() if getattr(alert, "departure_end", None) else None,
        "returnStart": alert.return_start.isoformat() if getattr(alert, "return_start", None) else None,
        "returnEnd": alert.return_end.isoformat() if getattr(alert, "return_end", None) else None,
        "alertId": getattr(alert, "id", None),
        "autoSearch": "1",
    }

    # Include passengers if present, so UI reflects alert settings
    passengers = getattr(alert, "passengers", None)
    if passengers is not None:
        try:
            qp["passengers"] = str(max(1, int(passengers)))
        except Exception:
            pass

    qp = {k: v for k, v in qp.items() if v is not None and v != ""}

    return f"{base}/SearchFlyyv?{urlencode(qp)}"

# =====================================
# END SECTION: LINK BUILDERS
# =====================================


# =======================================
# START SECTION: ALERT MATH HELPERS
# =======================================

def _derive_nights_label(alert, is_flex: bool) -> str:
    """
    Returns a friendly label like "7 nights" or "Flexible".
    """
    try:
        dep_start = getattr(alert, "departure_start", None)
        ret_start = getattr(alert, "return_start", None)
        if dep_start and ret_start:
            nights = max(1, (ret_start - dep_start).days)
            return f"{nights} nights"

        nights_val = getattr(alert, "nights", None)
        if nights_val is not None:
            return f"{max(1, int(nights_val))} nights"

        if not is_flex:
            return "Not set"
    except Exception:
        pass

    return "Flexible"


def _estimate_combinations_for_confirmation(alert) -> Optional[int]:
    """
    Produces a realistic combinations count for the confirmation email.
    Uses the same constraints as the scan:
    - window days
    - stay range
    - MAX_DATE_PAIRS and MAX_DATE_PAIRS_PER_ALERT caps (if available)
    """
    dep_start = getattr(alert, "departure_start", None)
    dep_end = getattr(alert, "departure_end", None)
    if not dep_start or not dep_end:
        return None

    try:
        window_days = (dep_end - dep_start).days + 1
        if window_days <= 0:
            return None
    except Exception:
        return None

    # Try to read stay range from alert, fallback to fixed nights derived from stored dates
    min_stay = getattr(alert, "minStayDays", None) or getattr(alert, "min_stay_days", None)
    max_stay = getattr(alert, "maxStayDays", None) or getattr(alert, "max_stay_days", None)

    if min_stay is None or max_stay is None:
        try:
            ret_start = getattr(alert, "return_start", None)
            if ret_start:
                fixed = max(1, (ret_start - dep_start).days)
                min_stay = fixed
                max_stay = fixed
        except Exception:
            min_stay = 1
            max_stay = 1

    try:
        min_stay = max(1, int(min_stay))
        max_stay = max(min_stay, int(max_stay))
    except Exception:
        min_stay = 1
        max_stay = 1

    # Match generate_date_pairs behaviour: ret <= departure_end
    theoretical = 0
    for stay in range(min_stay, max_stay + 1):
        theoretical += max(0, window_days - stay)

    if theoretical <= 0:
        return None

    # Apply caps if these helpers exist in runtime
    max_pairs_cap: Optional[int] = None
    try:
        # If you have config helpers, use them, otherwise fallback to typical defaults
        get_config_int = globals().get("get_config_int")
        if callable(get_config_int):
            cap_a = int(get_config_int("MAX_DATE_PAIRS", 60))
            cap_b = int(get_config_int("MAX_DATE_PAIRS_PER_ALERT", 40))
        else:
            cap_a = 60
            cap_b = 40

        hard_cap = int(globals().get("MAX_DATE_PAIRS_HARD", 60))
        max_pairs_cap = max(1, min(cap_a, hard_cap))
        if cap_b:
            max_pairs_cap = min(max_pairs_cap, cap_b)
    except Exception:
        max_pairs_cap = None

    if max_pairs_cap is not None:
        return min(theoretical, max_pairs_cap)

    return theoretical


def _passengers_label(alert) -> Tuple[int, str]:
    try:
        passengers = int(getattr(alert, "passengers", 1) or 1)
    except Exception:
        passengers = 1
    passengers = max(1, passengers)
    label = "1 passenger" if passengers == 1 else f"{passengers} passengers"
    return passengers, label

# =====================================
# END SECTION: ALERT MATH HELPERS
# =====================================


# =======================================
# START SECTION: ONE OFF ALERT EMAIL
# =======================================

def send_alert_email_for_alert(alert, cheapest, params) -> None:
    """
    One off alert email:
    single date pair, simple format.
    """
    _smtp_ready_or_raise()

    to_email = getattr(alert, "user_email", None)
    if not to_email:
        raise HTTPException(status_code=500, detail="Alert has no user_email")

    passengers, passengers_label = _passengers_label(alert)

    subject = (
        f"Flyyv Alert: {alert.origin} \u2192 {alert.destination} "
        f"from £{int(cheapest.price)}"
    )

    dep_dt = datetime.fromisoformat(cheapest.departureDate)
    ret_dt = datetime.fromisoformat(cheapest.returnDate)
    dep_label = dep_dt.strftime("%d %b %Y")
    ret_label = ret_dt.strftime("%d %b %Y")

    lines: List[str] = []
    lines.append(f"Flyyv Alert")
    lines.append(f"Route: {alert.origin} \u2192 {alert.destination}, {alert.cabin.title()} class")
    lines.append(f"Passengers: {passengers_label}")
    lines.append(f"Dates: {dep_label} to {ret_label}")
    lines.append("")
    lines.append(
        f"Best price found: £{int(cheapest.price)} "
        f"with {cheapest.airline} ({cheapest.airlineCode or ''})"
    )
    lines.append(f"Prices shown are for {passengers_label} (total).")
    lines.append("")
    lines.append("Open your results:")
    lines.append(build_alert_search_link(alert))
    lines.append("")
    lines.append("You are receiving this because you created a Flyyv price alert.")
    lines.append("To stop these alerts, delete the alert in your Flyyv profile.")

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = f"FLYYV <{ALERT_FROM_EMAIL}>"
    msg["To"] = to_email
    msg.set_content("\n".join(lines))

    _send_email_message(msg)

# =====================================
# END SECTION: ONE OFF ALERT EMAIL
# =====================================


# =======================================
# START SECTION: SMART ALERT SUMMARY EMAIL
# =======================================

def send_smart_alert_email(alert, options: List, params) -> None:
    """
    FlyyvFlex results email.
    Shows top cheapest date pairs and deep links per pair.
    """
    _smtp_ready_or_raise()

    to_email = getattr(alert, "user_email", None)
    if not to_email:
        raise HTTPException(status_code=500, detail="Alert has no user_email")

    threshold = getattr(alert, "max_price", None)
    origin = getattr(alert, "origin", "")
    destination = getattr(alert, "destination", "")
    cabin = getattr(alert, "cabin", "BUSINESS")

    passengers, passengers_label = _passengers_label(alert)

    # Remove temporary debug once confirmed stable
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

        min_price = min(prices)
        cheapest = min(flights, key=lambda o: o.price)

        flights_under: List = []
        if threshold is not None:
            flights_under = [o for o in flights if o.price <= float(threshold)]
            if flights_under:
                any_under = True

        flyyv_link = build_flyyv_link(params, dep_iso, ret_iso)

        pairs_summary.append(
            {
                "departureDate": dep_iso,
                "returnDate": ret_iso,
                "totalFlights": len(flights),
                "cheapestPrice": cheapest.price,
                "cheapestAirline": cheapest.airline,
                "flightsUnderThresholdCount": len(flights_under),
                "flyyvLink": flyyv_link,
                "minPrice": min_price,
            }
        )

    start_label = params.earliestDeparture.strftime("%d %b %Y")
    end_label = params.latestDeparture.strftime("%d %b %Y")

    nights_label = None
    try:
        if getattr(alert, "departure_start", None) and getattr(alert, "return_start", None):
            nights_val = max(1, (alert.return_start - alert.departure_start).days)
            nights_label = f"{nights_val}"
    except Exception:
        nights_label = None

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
    top_pairs_sorted = sorted(top_pairs, key=lambda x: x["cheapestPrice"])[:5]

    open_full_results_url = build_alert_search_link(alert)

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = f"FLYYV <{ALERT_FROM_EMAIL}>"
    msg["To"] = to_email

    lines: List[str] = []
    lines.append("FlyyvFlex Smart Search Alert")
    lines.append(f"Route: {origin} \u2192 {destination}, {str(cabin).title()} class")
    lines.append(f"Passengers: {passengers_label}")
    if nights_label:
        lines.append(f"Trip length: {nights_label} nights")
    lines.append(f"Date window: {start_label} to {end_label}")
    lines.append(f"Combinations checked: {combinations_checked}")
    if best_price_overall is not None:
        lines.append(f"Best price found: £{best_price_overall}")
    lines.append(f"Prices shown are for {passengers_label} (total).")
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

    msg.set_content("\n".join(lines))

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

            <div style="font-size:15px;line-height:1.6;color:#111827;margin:0 0 14px 0;">
              Based on a full scan of your <strong>{start_label} to {end_label}</strong> window
              {f" for <strong>{nights_label}-night</strong> trips" if nights_label else ""}.
            </div>

            <div style="margin:0 0 10px 0;font-size:13px;color:#6b7280;">
              Prices shown are for {passengers_label} (total).
            </div>

            <div style="margin:0 0 16px 0;">
              <span style="display:inline-block;padding:8px 12px;border-radius:999px;border:1px solid #d1fae5;background:#ecfdf5;font-size:13px;font-weight:800;margin-right:8px;margin-bottom:8px;">
                {cabin}
              </span>
              <span style="display:inline-block;padding:8px 12px;border-radius:999px;border:1px solid #e6e8ee;background:#f9fafb;font-size:13px;margin-right:8px;margin-bottom:8px;">
                {passengers_label}
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

# =====================================
# END SECTION: SMART ALERT SUMMARY EMAIL
# =====================================


# =======================================
# START SECTION: ALERT CONFIRMATION EMAIL
# =======================================

def send_alert_confirmation_email(alert) -> None:
    """
    Sent immediately when a user creates an alert.
    Confirms the alert is active.
    """
    if not (SMTP_USERNAME and SMTP_PASSWORD and ALERT_FROM_EMAIL):
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

    email_type_label = "FlyyvFlex Smart Search Alert" if is_flex else "Flyyv Alert"
    pill_type_label = "Smart price watch" if is_flex else "Price alert"

    dep_start_label = departure_start.strftime("%d %b %Y") if departure_start else "Not set"
    dep_end_label = departure_end.strftime("%d %b %Y") if departure_end else "Not set"
    dep_window_label = f"{dep_start_label} to {dep_end_label}"

    trip_length_label = _derive_nights_label(alert, is_flex)

    passengers, passengers_label = _passengers_label(alert)

    combinations_checked = _estimate_combinations_for_confirmation(alert)

    base = FRONTEND_BASE_URL.rstrip("/")
    results_url = (
        f"{base}/SearchFlyyv"
        f"?origin={origin}"
        f"&destination={destination}"
        f"&cabin={cabin}"
        f"&passengers={passengers}"
        f"&searchMode={'flexible' if is_flex else 'single'}"
        f"&departureStart={(departure_start.isoformat() if departure_start else '')}"
        f"&departureEnd={(departure_end.isoformat() if departure_end else '')}"
        f"&autoSearch=1"
    )

    if return_start:
        results_url += f"&returnStart={return_start.isoformat()}"
    if return_end:
        results_url += f"&returnEnd={return_end.isoformat()}"

    alert_id = getattr(alert, "id", None)

    subject = (
        f"{email_type_label}: {origin} \u2192 {destination} | "
        f"{dep_start_label} to {dep_end_label} | {trip_length_label}"
    )

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
    text_lines.append(f"Passengers: {passengers_label}")
    text_lines.append(f"Departure window: {dep_window_label}")
    text_lines.append(f"Trip length: {trip_length_label}")
    if combinations_checked:
        text_lines.append(f"Combinations checked: {combinations_checked}")
    if alert_id:
        text_lines.append(f"Alert ID: {alert_id}")
    text_lines.append(f"Prices shown are for {passengers_label} (total).")
    text_lines.append("")
    text_lines.append("View results:")
    text_lines.append(results_url)

    msg.set_content("\n".join(text_lines))

    html = f"""
    <html>
      <body style="margin:0;padding:0;background:#f6f7f9;font-family:Arial,Helvetica,sans-serif;">
        <div style="max-width:680px;margin:0 auto;padding:24px;">
          <div style="background:#ffffff;border:1px solid #e6e8ee;border-radius:14px;padding:26px;">
            <div style="font-size:14px;color:#6b7280;margin-bottom:10px;">{email_type_label}</div>

            <div style="font-size:28px;line-height:1.2;color:#111827;font-weight:700;margin:0 0 12px 0;">
              Your alert is active
            </div>

            <div style="font-size:16px;line-height:1.5;color:#111827;margin:0 0 14px 0;">
              We are watching <strong>{origin} \u2192 {destination}</strong> and will email you when prices match your alert conditions.
            </div>

            <div style="font-size:12px;color:#6b7280;margin:0 0 18px 0;">
              Prices shown are for {passengers_label} (total).
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

    msg.add_alternative(html, subtype="html")

    try:
        _send_email_message(msg)
    except Exception:
        pass

# =====================================
# END SECTION: ALERT CONFIRMATION EMAIL
# =====================================


# =======================================
# START SECTION: EARLY ACCESS WELCOME EMAIL
# =======================================

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

# =====================================
# END SECTION: EARLY ACCESS WELCOME EMAIL
# =====================================
