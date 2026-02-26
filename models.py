# =======================================
# SECTION: IMPORTS AND BASE
# =======================================

from datetime import datetime
from uuid import uuid4

from sqlalchemy import (
    Column,
    Integer,
    String,
    Boolean,
    DateTime,
    Date,
    Numeric,
    ForeignKey,
    Text,
)
from sqlalchemy.dialects.postgresql import UUID, JSONB

from db import Base


# =======================================
# SECTION: ADMIN CONFIG MODEL
# =======================================

class AdminConfig(Base):
    __tablename__ = "admin_config"

    id = Column(Integer, primary_key=True, index=True)
    key = Column(String(100), unique=True, nullable=False)
    value = Column(String(255), nullable=True)
    description = Column(String(255), nullable=True)

    # Global master switch for alerts
    alerts_enabled = Column(Boolean, nullable=False, default=True)


# =======================================
# SECTION: USER MODELS
# =======================================

class AppUser(Base):
    __tablename__ = "app_users"

    id = Column(Integer, primary_key=True, index=True)

    external_id = Column(String(100), unique=True, index=True, nullable=False)

    email = Column(String(255), index=True, nullable=False)
    first_name = Column(String(100), nullable=True)
    last_name = Column(String(100), nullable=True)
    country = Column(String(100), nullable=True)

    marketing_consent = Column(Boolean, default=None)
    source = Column(String(50), nullable=True)

    # Per user alerts switch
    email_alerts_enabled = Column(Boolean, nullable=False, default=True)

    # =======================================
    # SECTION: PLAN ENTITLEMENTS (v1)
    # =======================================
    plan_tier = Column(String(20), nullable=False, default="free")  # free | monthly | annual | tester | admin
    plan_active_alert_limit = Column(Integer, nullable=False, default=1)
    plan_max_departure_window_days = Column(Integer, nullable=False, default=15)
    plan_checks_per_day = Column(Integer, nullable=False, default=1)

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(
        DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
    )


# =======================================
# SECTION: EARLY ACCESS SUBSCRIBERS
# =======================================

class EarlyAccessSubscriber(Base):
    __tablename__ = "early_access_subscribers"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    email = Column(String(255), unique=True, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    is_notified = Column(Boolean, default=False, nullable=False)


# =======================================
# SECTION: ALERT MODELS
# =======================================

class Alert(Base):
    __tablename__ = "alerts"

    id = Column(String, primary_key=True, index=True)

    # Ownership
    user_email = Column(String, index=True, nullable=False)
    user_external_id = Column(String, nullable=True, index=True)

    # Search parameters
    origin = Column(String, nullable=False)
    destination = Column(String, nullable=False)
    cabin = Column(String, nullable=False)

    search_mode = Column(String(20), nullable=True, default="flexible")

    departure_start = Column(Date, nullable=False)
    departure_end = Column(Date, nullable=False)
    return_start = Column(Date, nullable=True)
    return_end = Column(Date, nullable=True)

    alert_type = Column(String, nullable=False)
    max_price = Column(Integer, nullable=True)

    # single = specific date pair
    # smart  = smart search / date range based
    mode = Column(String(32), nullable=False, default="single")

    passengers = Column(Integer, nullable=False, default=1)

    # Pricing and run tracking
    last_price = Column(Integer, nullable=True)
    last_run_at = Column(DateTime, nullable=True)

    # Duplicate guard, prevents double runs in the same tick
    last_checked_at = Column(DateTime, nullable=True)

    # Alert-only cache, reuse results for up to 8 hours
    cache_created_at = Column(DateTime, nullable=True)
    cache_expires_at = Column(DateTime, nullable=True)
    cache_payload_json = Column(Text, nullable=True)  # JSON string

    # Email notification tracking
    last_notified_at = Column(DateTime, nullable=True)
    last_notified_price = Column(Integer, nullable=True)

    times_sent = Column(Integer, nullable=False, default=0)

    is_active = Column(Boolean, nullable=False, default=True)

    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow)


class AlertRun(Base):
    __tablename__ = "alert_runs"

    id = Column(String, primary_key=True, index=True)
    alert_id = Column(String, ForeignKey("alerts.id"), nullable=False, index=True)

    run_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    price_found = Column(Integer, nullable=True)
    sent = Column(Boolean, nullable=False, default=False)
    reason = Column(Text, nullable=True)


# =======================================
# SECTION: SEARCH JOB PERSISTENCE
# Replaces in-memory JOBS / JOB_RESULTS.
# Survives container restarts.
# =======================================

class SearchJob(Base):
    __tablename__ = "search_jobs"

    id = Column(String(255), primary_key=True, index=True)
    status = Column(String(50), nullable=False, default="pending")  # pending|running|completed|failed|cancelled
    params_json = Column(JSONB, nullable=False)
    total_pairs = Column(Integer, nullable=False, default=0)
    processed_pairs = Column(Integer, nullable=False, default=0)
    error = Column(Text, nullable=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow)


class SearchResult(Base):
    __tablename__ = "search_results"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    job_id = Column(String(255), ForeignKey("search_jobs.id"), nullable=False, index=True)
    offer_json = Column(JSONB, nullable=False)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)


# =======================================
# SECTION: CLICK TRACKING
# Logs every outbound booking click for analytics.
# =======================================

class OfferClick(Base):
    __tablename__ = "offer_clicks"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    clicked_at = Column(DateTime, nullable=False, default=datetime.utcnow)

    user_external_id = Column(String(255), nullable=True, index=True)
    offer_id = Column(String(512), nullable=True)
    airline_code = Column(String(10), nullable=True)
    price = Column(Numeric(10, 2), nullable=True)
    currency = Column(String(10), nullable=True)
    origin = Column(String(10), nullable=True)
    destination = Column(String(10), nullable=True)
    departure_date = Column(Date, nullable=True)
    return_date = Column(Date, nullable=True)
    cabin = Column(String(50), nullable=True)
    stops = Column(Integer, nullable=True)

    # "search" | "alert_email" | "preview"
    source = Column(String(50), nullable=True)
    job_id = Column(String(255), nullable=True)
    alert_id = Column(String(255), nullable=True)

    destination_url = Column(Text, nullable=True)
    provider = Column(String(50), nullable=False, default="ttn")
    redirect_followed = Column(Boolean, nullable=False, default=True)
