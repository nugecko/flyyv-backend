import os
import smtplib
from email.message import EmailMessage
from datetime import datetime
from typing import List, Dict, Tuple, Any
from urllib.parse import urlencode

from fastapi import HTTPException

# =======================================
# SMTP CONFIG AND CONSTANTS
# =======================================

SMTP_HOST = os.environ.get("SMTP_HOST", "smtp.smtp2go.com")
SMTP_PORT = int(os.environ.get("SMTP_PORT", "587"))
SMTP_USERNAME = os.environ.get("SMTP_USERNAME")
SMTP_PASSWORD = os.environ.get("SMTP_PASSWORD")
ALERT_FROM_EMAIL = os.environ.get("ALERT_FROM_EMAIL", "alert@flyyv.com")

FRONTEND_BASE_URL = os.environ.get("FRONTEND_BASE_URL", "https://flyyv.com")


# =======================================
# HELPER LINK BUILDERS
# =======================================

def build_date_pair_link(params, departure: str, return_date: str) -> str:
    """
    Drilldown into a single exact date pair.
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

    alert_id = getattr(params, "id", None)
    if alert_id:
        qp["alertId"] = str(alert_id)

    return f"{base}/SearchFlyyv?{urlencode(qp)}"


def build_flexible_results_link(alert) -> str:
    """
    Recreate the original smart alert window.
    """
    base = FRONTEND_BASE_URL.rstrip("/")

    params = {
        "origin": alert.origin,
        "destination": alert.destination,
        "cabin": alert.cabin,
        "searchMode": "flexible",
        "departureStart": alert.departure_start.isoformat(),
        "departureEnd": alert.departure_end.isoformat(),
        "autoSearch": "1",
        "alertId": getattr(alert, "id", None),
    }

    if alert.departure_start and alert.return_start:
        nights = max(1, (alert.return_start - alert.departure_start).days)
        params["nights"] = str(nights)

    return f"{base}/SearchFlyyv?{urlencode(params)}"


# =======================================
# ONE OFF SINGLE DATE ALERT EMAIL
# =======================================

def send_alert_email_for_alert(alert, cheapest, params) -> None:
    if not (SMTP_USERNAME and SMTP_PASSWORD and ALERT_FROM_EMAIL):
        raise HTTPException(status_code=500, detail="SMTP not configured")

    subject = f"Flyyv Alert: {alert.origin} → {alert.destination} from £{int(cheapest.price)}"

    dep = datetime.fromisoformat(cheapest.departureDate).strftime("%d %b %Y")
    ret = datetime.fromisoformat(cheapest.returnDate).strftime("%d %b %Y")

    body = (
        f"Flyyv Alert\n\n"
        f"Route: {alert.origin} → {alert.destination}\n"
        f"Cabin: {alert.cabin}\n"
        f"Dates: {dep} to {ret}\n\n"
        f"Best price found: £{int(cheapest.price)} with {cheapest.airline}\n\n"
        f"View results:\n{FRONTEND_BASE_URL}\n\n"
        f"You are receiving this email because you created a Flyyv alert."
    )

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = f"FLYYV <{ALERT_FROM_EMAIL}>"
    msg["To"] = alert.user_email
    msg.set_content(body)

    with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
        server.starttls()
        server.login(SMTP_USERNAME, SMTP_PASSWORD)
        server.send_message(msg)


# =======================================
# SMART ALERT SUMMARY EMAIL
# =======================================

def send_smart_alert_email(alert, options: List, params) -> None:
    if not (SMTP_USERNAME and SMTP_PASSWORD and ALERT_FROM_EMAIL):
        raise HTTPException(status_code=500, detail="SMTP not configured")

    origin = alert.origin
    destination = alert.destination

    grouped: Dict[Tuple[str, str], List] = {}
    for opt in options:
        grouped.setdefault((opt.departureDate, opt.returnDate), []).append(opt)

    summaries = []
    for (dep, ret), flights in grouped.items():
        cheapest = min(flights, key=lambda o: o.price)
        summaries.append(
            {
                "departure": dep,
                "return": ret,
                "price": int(cheapest.price),
                "airline": cheapest.airline,
                "link": build_date_pair_link(alert, dep, ret),
            }
        )

    summaries = sorted(summaries, key=lambda x: x["price"])[:5]

    start_label = alert.departure_start.strftime("%d %b %Y")
    end_label = alert.departure_end.strftime("%d %b %Y")
    nights = max(1, (alert.return_start - alert.departure_start).days)

    best_price = summaries[0]["price"] if summaries else None

    subject = (
        f"FlyyvFlex Alert: {origin} → {destination} from £{best_price}"
        if best_price
        else f"FlyyvFlex Alert: {origin} → {destination}"
    )

    open_full_results_url = build_flexible_results_link(alert)

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = f"FLYYV <{ALERT_FROM_EMAIL}>"
    msg["To"] = alert.user_email

    text_lines = [
        f"FlyyvFlex Smart Search Alert",
        "",
        f"Route: {origin} → {destination}",
        f"Window: {start_label} to {end_label}",
        f"Trip length: {nights} nights",
        "",
        "Top cheapest dates:",
        "",
    ]

    for s in summaries:
        dep = datetime.fromisoformat(s["departure"]).strftime("%d %b")
        ret = datetime.fromisoformat(s["return"]).strftime("%d %b")
        text_lines.append(f"£{s['price']} | {dep} → {ret} | {s['airline']}")
        text_lines.append(s["link"])
        text_lines.append("")

    text_lines.append("Open full results:")
    text_lines.append(open_full_results_url)

    msg.set_content("\n".join(text_lines))

    rows_html = ""
    for s in summaries:
        dep = datetime.fromisoformat(s["departure"]).strftime("%d %b %Y")
        ret = datetime.fromisoformat(s["return"]).strftime("%d %b %Y")
        rows_html += f"""
        <tr>
          <td style="padding:14px;border:1px solid #e6e8ee;border-radius:12px;margin-bottom:10px;display:block;">
            <strong>{origin} → {destination}</strong><br>
            {dep} to {ret}<br>
            <strong>£{s["price"]}</strong><br>
            Our pick: {s["airline"]}<br>
            <a href="{s["link"]}">View flight</a>
          </td>
        </tr>
        """

    html = f"""
    <html>
      <body style="font-family:Arial;background:#f6f7f9;padding:24px;">
        <div style="max-width:680px;margin:auto;background:#ffffff;padding:26px;border-radius:14px;">
          <h2>Top deals for {origin} → {destination}</h2>
          <p>
            Based on a full scan of your <strong>{start_label} to {end_label}</strong>
            window for <strong>{nights}-night</strong> trips.
          </p>

          <p><strong>Best price found: £{best_price}</strong></p>

          <table width="100%" cellpadding="0" cellspacing="0">
            {rows_html}
          </table>

          <p>
            <a href="{open_full_results_url}"
               style="display:inline-block;background:#111827;color:#ffffff;
               padding:12px 16px;border-radius:10px;text-decoration:none;">
              Open full results
            </a>
          </p>

          <p style="font-size:12px;color:#6b7280;">
            You are receiving this email because you created a FlyyvFlex Smart Search Alert.
          </p>
        </div>
      </body>
    </html>
    """

    msg.add_alternative(html, subtype="html")

    with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
        server.starttls()
        server.login(SMTP_USERNAME, SMTP_PASSWORD)
        server.send_message(msg)
