"""
main.py

Flyyv backend entry point.
Responsible only for: creating the FastAPI app, CORS, and mounting routers.
All logic lives in routers/, services/, providers/.
"""

import smtplib
from email.message import EmailMessage

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config import ALERT_FROM_EMAIL, ALERT_TO_EMAIL, SMTP_HOST, SMTP_PASSWORD, SMTP_PORT, SMTP_USERNAME
from db import engine, Base
from routers import search, alerts, users, admin, ttn, clicks
from early_access import router as early_access_router
import models  # noqa: F401 — ensure all models are registered before create_all

app = FastAPI()


@app.on_event("startup")
def on_startup():
    Base.metadata.create_all(bind=engine)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(search.router)
app.include_router(alerts.router)
app.include_router(users.router)
app.include_router(admin.router)
app.include_router(ttn.router)
app.include_router(clicks.router)
app.include_router(early_access_router)


# =====================================================================
# SECTION: SMTP TEST HELPER (used by /test-email-alert route in admin.py)
# =====================================================================

def _smtp_send_test() -> None:
    if not (SMTP_USERNAME and SMTP_PASSWORD and ALERT_TO_EMAIL):
        from fastapi import HTTPException
        raise HTTPException(status_code=500, detail="SMTP settings are not fully configured on the server")

    msg = EmailMessage()
    msg["Subject"] = "Flyyv test alert email"
    msg["From"] = ALERT_FROM_EMAIL
    msg["To"] = ALERT_TO_EMAIL
    msg.set_content("This is a test Flyyv alert sent via SMTP2Go.\n\nIf you are reading this, SMTP is working.")

    try:
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
            server.starttls()
            server.login(SMTP_USERNAME, SMTP_PASSWORD)
            server.send_message(msg)
    except Exception as e:
        from fastapi import HTTPException
        raise HTTPException(status_code=500, detail=f"Failed to send test email: {e}")
