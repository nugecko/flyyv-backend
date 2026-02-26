"""
config.py

Single source of truth for:
- Environment variable reads
- Admin config DB helpers
- Alert toggle logic
- Plan tier defaults

Nothing here should contain route handlers or business logic beyond config resolution.
"""

import os
from typing import Optional

from sqlalchemy.orm import Session


# =====================================================================
# SECTION: ENV VARS
# =====================================================================

ADMIN_API_TOKEN = os.getenv("ADMIN_API_TOKEN")

# Duffel (legacy / direct-only fetches)
DUFFEL_ACCESS_TOKEN = os.getenv("DUFFEL_ACCESS_TOKEN")
DUFFEL_API_BASE = "https://api.duffel.com"
DUFFEL_API_TOKEN = os.getenv("DUFFEL_API_TOKEN") or DUFFEL_ACCESS_TOKEN

# Flight provider routing
# Set FLIGHT_PROVIDER=flightapi to switch from TTN to Flight API.
FLIGHT_PROVIDER = os.getenv("FLIGHT_PROVIDER", "ttn").lower().strip()
FLIGHTAPI_KEY = os.getenv("FLIGHTAPI_KEY", "")

# TTN
TTN_BASE_URL = "https://v2.api.tickets.ua"
TTN_API_KEY = os.getenv("TTN_API_KEY", "")
TTN_AUTH_KEY = os.getenv("TTN_AUTH_KEY", "")

# SMTP / alerts
SMTP_HOST = os.getenv("SMTP_HOST", "mail-eu.smtp2go.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", "2525"))
SMTP_USERNAME = os.getenv("SMTP_USERNAME")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD")
ALERT_FROM_EMAIL = os.getenv("ALERT_FROM_EMAIL", "price-alert@flyyv.com")
ALERT_TO_EMAIL = os.getenv("ALERT_TO_EMAIL")

FRONTEND_BASE_URL = os.getenv("FRONTEND_BASE_URL", "https://app.flyyv.com")

# Price watch (legacy utility)
WATCH_ORIGIN = os.getenv("WATCH_ORIGIN", "LON")
WATCH_DESTINATION = os.getenv("WATCH_DESTINATION", "TLV")
WATCH_START_DATE = os.getenv("WATCH_START_DATE")
WATCH_END_DATE = os.getenv("WATCH_END_DATE")
WATCH_STAY_NIGHTS = int(os.getenv("WATCH_STAY_NIGHTS", "7"))
WATCH_MAX_PRICE = float(os.getenv("WATCH_MAX_PRICE", "720"))

ALERTS_ENABLED = os.getenv("ALERTS_ENABLED", "true").lower() == "true"

# Search caps (hard limits enforced in code, not overridable by admin config)
MAX_OFFERS_PER_PAIR_HARD = 300
MAX_OFFERS_TOTAL_HARD = 4000
MAX_DATE_PAIRS_HARD = 60
SYNC_PAIR_THRESHOLD = 10
PARALLEL_WORKERS = 6

BASE44_WEBHOOK_SECRET = os.getenv("BASE44_WEBHOOK_SECRET")


# =====================================================================
# SECTION: PLAN TIER DEFAULTS
# Single source of truth for all plan entitlements.
# Used by /user-sync, /admin/sync-user-tier, and /base44/user-webhook.
# =====================================================================

PLAN_DEFAULTS = {
    "free": {
        "plan_tier": "free",
        "plan_active_alert_limit": 1,
        "plan_max_departure_window_days": 7,
        "plan_checks_per_day": 3,
    },
    "gold": {
        "plan_tier": "gold",
        "plan_active_alert_limit": 3,
        "plan_max_departure_window_days": 14,
        "plan_checks_per_day": 6,
    },
    "platinum": {
        "plan_tier": "platinum",
        "plan_active_alert_limit": 10,
        "plan_max_departure_window_days": 30,
        "plan_checks_per_day": 12,
    },
    "tester": {
        "plan_tier": "tester",
        "plan_active_alert_limit": 10_000,
        "plan_max_departure_window_days": 365,
        "plan_checks_per_day": 10_000,
    },
    "admin": {
        "plan_tier": "admin",
        "plan_active_alert_limit": 10_000,
        "plan_max_departure_window_days": 365,
        "plan_checks_per_day": 10_000,
    },
}

ALLOWED_TIERS = set(PLAN_DEFAULTS.keys())


# =====================================================================
# SECTION: ADMIN CONFIG DB HELPERS
# Read runtime configuration values stored in admin_config table.
# =====================================================================

def _get_config_row(db: Session, key: str):
    from models import AdminConfig
    return db.query(AdminConfig).filter(AdminConfig.key == key).first()


def get_config_str(key: str, default_value: Optional[str] = None) -> Optional[str]:
    """Read a config value from admin_config as string."""
    from db import SessionLocal
    db = SessionLocal()
    try:
        row = _get_config_row(db, key)
        if not row or row.value is None:
            return default_value
        return str(row.value)
    finally:
        db.close()


def get_config_int(key: str, default_value: int) -> int:
    """Read a config value from admin_config and cast to int."""
    raw = get_config_str(key, None)
    if raw is None:
        return default_value
    try:
        return int(raw)
    except ValueError:
        return default_value


def get_config_bool(key: str, default_value: bool) -> bool:
    """Read a config value from admin_config and cast to bool."""
    raw = get_config_str(key, None)
    if raw is None:
        return default_value
    s = str(raw).strip().lower()
    if s in ("1", "true", "yes", "y", "on"):
        return True
    if s in ("0", "false", "no", "n", "off"):
        return False
    return default_value


# =====================================================================
# SECTION: ALERT TOGGLE HELPERS
# =====================================================================

def master_alerts_enabled() -> bool:
    """Hard master switch controlled by ALERTS_ENABLED env var."""
    value = os.getenv("ALERTS_ENABLED", "true")
    return value.lower() == "true"


def alerts_globally_enabled(db: Session) -> bool:
    """Global switch stored in admin_config key = 'GLOBAL_ALERTS'."""
    from models import AdminConfig
    config = db.query(AdminConfig).filter(AdminConfig.key == "GLOBAL_ALERTS").first()
    if not config:
        return True
    if not hasattr(config, "alerts_enabled"):
        return True
    return bool(config.alerts_enabled)


def user_allows_alerts(user) -> bool:
    """Per-user toggle, defaults to True if the column is missing."""
    if not hasattr(user, "email_alerts_enabled"):
        return True
    return bool(user.email_alerts_enabled)


def should_send_alert(db: Session, user) -> bool:
    """
    Combined logic:
    1. Environment master toggle must be ON
    2. Global admin_config toggle must be ON
    3. User toggle must be ON
    """
    if not master_alerts_enabled():
        return False
    if not alerts_globally_enabled(db):
        return False
    if not user_allows_alerts(user):
        return False
    return True
