import os
import smtplib
from email.message import EmailMessage
from datetime import datetime
from typing import List, Dict, Tuple, Any

from fastapi import HTTPException
from urllib.parse import urlencode

# =======================================
# SECTION: SMTP CONFIG AND CONSTANTS
# =======================================

# Basic SMTP config, read from environment
SMTP_HOST = os.environ.get("SMTP_HOST", "smtp.smtp2go.com")
SMTP_PORT = int(os.environ.get("SMTP_PORT", "587"))
SMTP_USERNAME = os.environ.get("SMTP_USERNAME")
SMTP_PASSWORD = os.environ.get("SMTP_PASSWORD")
ALERT_FROM_EMAIL = os.environ.get("ALERT_FROM_EMAIL", "alert@flyyv.com")

# FRONTEND_BASE_URL must come from environment
FRONTEND_BASE_URL = os.environ.get("FRONTEND_BASE_URL", "https://flyyv.com")


# =======================================
# SECTION: ONE OFF ALERT EMAIL
# =======================================

# ===== START ONE OFF ALERT EMAIL =====
def send_alert_email_for_alert(alert, cheapest, params) -> None:
    """
    One off alert email:
    single date pair, simple format.
    """
    if not (SMTP_USERNAME and SMTP_PASSWORD and ALERT_FROM_EMAIL):
        raise HTTPException(
            status_code=500,
            detail="SMTP settings are not fully configured on the server",
        )

    to_email = alert.user_email
    if not to_email:
        raise HTTPException(status_code=500, detail="Alert has no user_email")

    subject = (
        f"Flyyv Alert: {alert.origin} \u2192 {alert.destination} "
        f"from £{int(cheapest.price)}"
    )

    dep_dt = datetime.fromisoformat(cheapest.departureDate)
    ret_dt = datetime.fromisoformat(cheapest.returnDate)
    dep_label = dep_dt.strftime("%d-%m-%Y")
    ret_label = ret_dt.strftime("%d-%m-%Y")

    lines: List[str] = []

    lines.append(
        f"Route: {alert.origin} \u2192 {alert.destination}, {alert.cabin.title()} class"
    )
    lines.append(f"Dates: {dep_label} to {ret_label}")
    lines.append("")
    lines.append(
        f"Cheapest fare found: £{int(cheapest.price)} "
        f"with {cheapest.airline} ({cheapest.airlineCode or ''})"
    )
    lines.append("")
    lines.append("To view this alert and explore more dates, go to your Flyyv dashboard:")
    lines.append(FRONTEND_BASE_URL)
    lines.append("")
    lines.append("You are receiving this because you created a Flyyv price alert.")
    lines.append("To stop these alerts, delete the alert in your Flyyv profile.")

    body = "\n".join(lines)

    msg = EmailMessage()
    msg["Subject"] = subject
    # From header with display name
    msg["From"] = f"FLYYV <{ALERT_FROM_EMAIL}>"
    msg["To"] = to_email
    msg.set_content(body)

    with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
        server.starttls()
        server.login(SMTP_USERNAME, SMTP_PASSWORD)
        server.send_message(msg)
# ===== END ONE OFF ALERT EMAIL =====

# =======================================
# SECTION: HELPER LINK BUILDERS
# =======================================

# ===== START HELPER LINK BUILDERS =====
def build_flyyv_link(params, departure: str, return_date: str) -> str:
    base = FRONTEND_BASE_URL.rstrip("/")
    return (
        f"{base}/?origin={params.origin}"
        f"&destination={params.destination}"
        f"&departure={departure}"
        f"&return={return_date}"
        f"&cabin={params.cabin}"
        f"&passengers={params.passengers}"
    )

def build_alert_search_link(alert) -> str:
    base = FRONTEND_BASE_URL.rstrip("/")

    params = {
        "origin": alert.origin,
        "destination": alert.destination,
        "cabin": alert.cabin,
                # These fields help the frontend decide whether to run FlyyvFlex or single
        "searchMode": ("flexible" if (getattr(alert, "mode", None) == "smart" or getattr(alert, "search_mode", None) == "flexible") else "single"),
        "nights": None,  # set below if we can derive it
        "mode": getattr(alert, "mode", None),
        "search_mode": getattr(alert, "search_mode", None),
        "departureStart": alert.departure_start.isoformat() if getattr(alert, "departure_start", None) else None,
        "departureEnd": alert.departure_end.isoformat() if getattr(alert, "departure_end", None) else None,
        "returnStart": alert.return_start.isoformat() if getattr(alert, "return_start", None) else None,
        "returnEnd": alert.return_end.isoformat() if getattr(alert, "return_end", None) else None,
        "alertId": getattr(alert, "id", None),
        "autoSearch": "1",
    }

    # remove None values
    params = {k: v for k, v in params.items() if v is not None}
        # If we can derive a single trip length, include it, this helps /SearchFlyyv choose FlyyvFlex mode
    try:
        if getattr(alert, "departure_start", None) and getattr(alert, "return_start", None):
            nights = max(1, (alert.return_start - alert.departure_start).days)
            params["nights"] = str(nights)
    except Exception:
        pass


    return f"{base}/SearchFlyyv?{urlencode(params)}"
    
# ===== END HELPER LINK BUILDERS =====

# =======================================
# SECTION: SMART ALERT SUMMARY EMAIL
# =======================================

# ===== START SMART ALERT SUMMARY EMAIL =====
def send_smart_alert_email(alert, options: List, params) -> None:
    """
    FlyyvFlex Monitor email:
    scans multiple date pairs in a flexible window and produces a summary message.
    """
    if not (SMTP_USERNAME and SMTP_PASSWORD and ALERT_FROM_EMAIL):
        raise HTTPException(
            status_code=500,
            detail="SMTP settings are not fully configured on the server",
        )

    to_email = alert.user_email
    if not to_email:
        raise HTTPException(status_code=500, detail="Alert has no user_email")

    threshold = alert.max_price
    origin = alert.origin
    destination = alert.destination

    # Group options by (departureDate, returnDate)
    grouped: Dict[Tuple[str, str], List] = {}
    for opt in options:
        key = (opt.departureDate, opt.returnDate)
        grouped.setdefault(key, []).append(opt)

    # Sort pairs by date
    sorted_keys = sorted(grouped.keys())

    any_under = False
    pairs_summary: List[Dict[str, Any]] = []

    for dep_iso, ret_iso in sorted_keys:
        flights = grouped[(dep_iso, ret_iso)]
        prices = [o.price for o in flights]
        if not prices:
            continue

        min_price = min(prices)
        max_price = max(prices)
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
                "minPrice": min_price,
                "maxPrice": max_price,
                "cheapestPrice": cheapest.price,
                "cheapestAirline": cheapest.airline,
                "flightsUnderThresholdCount": len(flights_under),
                "flyyvLink": flyyv_link,
            }
        )

    # Date labels
    start_label = params.earliestDeparture.strftime("%d %B %Y")
    end_label = params.latestDeparture.strftime("%d %B %Y")

    # Nights label, based on actual trip length of the cheapest overall option
    nights_text = None
    if options:
        first_opt = min(options, key=lambda o: o.price)
        try:
            dep_dt_first = datetime.fromisoformat(first_opt.departureDate)
            ret_dt_first = datetime.fromisoformat(first_opt.returnDate)
            nights_val = max(1, (ret_dt_first - dep_dt_first).days)
            nights_text = f"{nights_val}"
        except Exception:
            nights_text = None

    combinations_checked = len(pairs_summary)

    # Cheapest price across all options
    cheapest_price_overall = None
    if options:
        cheapest_overall = min(options, key=lambda o: o.price)
        try:
            cheapest_price_overall = int(cheapest_overall.price)
        except Exception:
            cheapest_price_overall = None

    # Subject
    if cheapest_price_overall is not None:
        subject = (
            f"FlyyvFlex Alert: {origin} \u2192 {destination} "
            f"from £{cheapest_price_overall}"
        )
    elif threshold is not None and any_under:
        subject = (
            f"FlyyvFlex Alert: {origin} \u2192 {destination} "
            f"fares under £{int(threshold)}"
        )
    elif threshold is not None:
        subject = (
            f"FlyyvFlex Alert: {origin} \u2192 {destination} "
            f"no fares under £{int(threshold)}"
        )
    else:
        subject = f"FlyyvFlex Alert: {origin} \u2192 {destination} update"

    lines: List[str] = []

    # Header
    lines.append(
        f"FlyyvFlex Monitor: {origin} \u2192 {destination}, "
        f"{alert.cabin.title()} class"
    )
    if nights_text:
        lines.append(f"Nights: {nights_text}")
    lines.append(f"Date window: {start_label} to {end_label}")
    lines.append(f"Possible combinations checked: {combinations_checked}")
    lines.append("")

    # Intro
    if threshold is not None:
        lines.append(f"Max budget: £{int(threshold)}")
    lines.append("")
    lines.append(
        "We scanned all matching date combinations in your selected window "
        "and highlighted the best fares available right now."
    )
    lines.append("")

    # Select top options
    top_pairs = [
        p
        for p in pairs_summary
        if p.get("totalFlights", 0) > 0 and p.get("cheapestPrice") is not None
    ]

    if not top_pairs:
        lines.append("No flights were found in this window in the latest scan.")
    else:
        MAX_RESULTS = 10
        top_pairs_sorted = sorted(
            top_pairs,
            key=lambda x: x["cheapestPrice"],
        )[:MAX_RESULTS]

        lines.append("Top flight deals in your window:")
        lines.append("")

        for p in top_pairs_sorted:
            dep_dt = datetime.fromisoformat(p["departureDate"])
            ret_dt = datetime.fromisoformat(p["returnDate"])
            dep_label = dep_dt.strftime("%d %b")
            ret_label = ret_dt.strftime("%d %b")

            price_label = int(p["cheapestPrice"])
            airline_label = p.get("cheapestAirline") or "Multiple airlines"

            line = (
                f"£{price_label}, {dep_label} \u2192 {ret_label}, "
                f"{airline_label}"
            )

            if threshold is not None and float(price_label) <= float(threshold):
                line += "  (within your limit)"

            lines.append(line)

    lines.append("")
    lines.append("View and manage your alerts:")
    lines.append(FRONTEND_BASE_URL)
    lines.append("")
    lines.append(
        "You are receiving this email because you created a FlyyvFlex Monitor alert."
    )
    lines.append("To stop these alerts, delete the alert in your Flyyv profile.")

    body = "\n".join(lines)

    msg = EmailMessage()
    msg["Subject"] = subject
    # From header with display name
    msg["From"] = f"FLYYV <{ALERT_FROM_EMAIL}>"
    msg["To"] = to_email
    msg.set_content(body)

    with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
        server.starttls()
        server.login(SMTP_USERNAME, SMTP_PASSWORD)
        server.send_message(msg)
# ===== END SMART ALERT SUMMARY EMAIL =====

# =======================================
# SECTION: ALERT CONFIRMATION EMAIL
# =======================================

# ===== START ALERT CONFIRMATION EMAIL =====
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

    # Date labels
    dep_start_label = departure_start.strftime("%d %b %Y") if departure_start else "Not set"
    dep_end_label = departure_end.strftime("%d %b %Y") if departure_end else "Not set"
    dep_window_label = f"{dep_start_label} to {dep_end_label}"

        # Trip length label
    trip_length_label = "Flexible"
    try:
        if departure_start and return_start:
            nights = max(1, (return_start - departure_start).days)
            trip_length_label = f"{nights} nights"
        elif getattr(alert, "nights", None):
            trip_length_label = f"{int(getattr(alert, 'nights'))} nights"
        elif not is_flex:
            trip_length_label = "Not set"
    except Exception:
        trip_length_label = "Flexible"

    # Build results link using the /SearchFlyyv prefix
    base = FRONTEND_BASE_URL.rstrip("/")
    results_url = (
        f"{base}/SearchFlyyv"
        f"?origin={origin}"
        f"&destination={destination}"
        f"&cabin={cabin}"
        f"&searchMode={'flexible' if is_flex else 'single'}"
        f"&departureStart={(departure_start.isoformat() if departure_start else '')}"
        f"&departureEnd={(departure_end.isoformat() if departure_end else '')}"
        f"&autoSearch=1"
    )

    if return_start:
        results_url += f"&returnStart={return_start.isoformat()}"
    if return_end:
        results_url += f"&returnEnd={return_end.isoformat()}"

    subject = f"{email_type_label}: {origin} \u2192 {destination}"

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = f"FLYYV <{ALERT_FROM_EMAIL}>"
    msg["To"] = to_email

    # Plain text fallback
    text_body = (
        f"{email_type_label}\n\n"
        "Your alert is active.\n\n"
        f"Route: {origin} \u2192 {destination}\n"
        f"Cabin: {cabin}\n"
        f"Departure window: {dep_window_label}\n"
        f"Trip length: {trip_length_label}\n\n"
        "View results:\n"
        f"{results_url}\n"
    )
    msg.set_content(text_body)

    # Light HTML version
    html = f"""
    <html>
      <body style="margin:0;padding:0;background:#f6f7f9;font-family:Arial,Helvetica,sans-serif;">
        <div style="max-width:680px;margin:0 auto;padding:24px;">
          <div style="background:#ffffff;border:1px solid #e6e8ee;border-radius:14px;padding:26px;">
            <div style="font-size:14px;color:#6b7280;margin-bottom:10px;">{email_type_label}</div>

            <div style="font-size:28px;line-height:1.2;color:#111827;font-weight:700;margin:0 0 12px 0;">
              Your alert is active
            </div>

            <div style="font-size:16px;line-height:1.5;color:#111827;margin:0 0 16px 0;">
              We are watching <strong>{origin} \u2192 {destination}</strong> and will email you when prices match your alert conditions.
            </div>

            <div style="margin:0 0 18px 0;">
              <span style="display:inline-block;padding:8px 12px;border-radius:999px;border:1px solid #e6e8ee;background:#f9fafb;font-size:13px;margin-right:8px;">
                {origin} \u2192 {destination}
              </span>
              <span style="display:inline-block;padding:8px 12px;border-radius:999px;border:1px solid #d1fae5;background:#ecfdf5;font-size:13px;font-weight:700;margin-right:8px;">
                {cabin}
              </span>
              <span style="display:inline-block;padding:8px 12px;border-radius:999px;border:1px solid #e6e8ee;background:#f9fafb;font-size:13px;margin-right:8px;">
                {pill_type_label}
              </span>
              <span style="display:inline-block;padding:8px 12px;border-radius:999px;border:1px solid #e6e8ee;background:#f9fafb;font-size:13px;margin-right:8px;">
                {dep_window_label}
              </span>
              <span style="display:inline-block;padding:8px 12px;border-radius:999px;border:1px solid #e6e8ee;background:#f9fafb;font-size:13px;">
                {trip_length_label}
              </span>
            </div>

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
              You are receiving this email because you created a Flyyv alert.
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
# ===== END ALERT CONFIRMATION EMAIL =====

# =======================================
# SECTION: EARLY ACCESS WELCOME EMAIL
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

# ===== END SECTION: EARLY ACCESS WELCOME EMAIL =====

