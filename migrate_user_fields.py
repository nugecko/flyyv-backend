"""
migrate_user_fields.py
----------------------
Adds is_trial, trial_expires_at, is_complimentary to app_users table.
Also migrates any legacy "free" tier users to "gold".

Run once on the server:
  dokku run flyyv python migrate_user_fields.py

Safe to run multiple times — uses ADD COLUMN IF NOT EXISTS.
"""

from datetime import datetime, timedelta, timezone
from sqlalchemy import text
from db import SessionLocal, engine


def run():
    print("[migrate] Starting migration...")

    with engine.connect() as conn:
        # Add new columns (safe / idempotent)
        conn.execute(text("""
            ALTER TABLE app_users
            ADD COLUMN IF NOT EXISTS is_trial BOOLEAN NOT NULL DEFAULT TRUE;
        """))
        conn.execute(text("""
            ALTER TABLE app_users
            ADD COLUMN IF NOT EXISTS trial_expires_at TIMESTAMP NULL;
        """))
        conn.execute(text("""
            ALTER TABLE app_users
            ADD COLUMN IF NOT EXISTS is_complimentary BOOLEAN NOT NULL DEFAULT FALSE;
        """))
        conn.commit()
        print("[migrate] Columns added (or already existed)")

    db = SessionLocal()
    try:
        from models import AppUser

        # 1. Migrate legacy "free" tier users → "gold" + is_trial=True
        #    Set trial_expires_at = created_at + 7 days (retroactive grace period)
        free_users = db.query(AppUser).filter(AppUser.plan_tier == "free").all()
        print(f"[migrate] Found {len(free_users)} legacy 'free' tier users to migrate")
        for u in free_users:
            u.plan_tier = "gold"
            u.plan_active_alert_limit = 2
            u.plan_max_departure_window_days = 30
            u.plan_checks_per_day = 1
            u.is_trial = True
            # Grace: trial expires 7 days from account creation
            if u.created_at:
                u.trial_expires_at = u.created_at + timedelta(days=7)
            else:
                u.trial_expires_at = datetime.now(timezone.utc) + timedelta(days=7)
            print(f"  → {u.email}: free → gold (trial expires {u.trial_expires_at.date()})")

        # 2. Tester and admin users: not on trial, not complimentary
        internal_users = db.query(AppUser).filter(
            AppUser.plan_tier.in_(["tester", "admin"])
        ).all()
        for u in internal_users:
            u.is_trial = False
            u.trial_expires_at = None
            u.is_complimentary = False
            print(f"  → {u.email}: {u.plan_tier} — trial cleared")

        # 3. Existing gold/platinum users with no trial info:
        #    Assume they're on a real paid plan (is_trial=False)
        #    (They were set up manually before this system existed)
        paid_users = db.query(AppUser).filter(
            AppUser.plan_tier.in_(["gold", "platinum"]),
            AppUser.is_trial == True,
            AppUser.trial_expires_at == None,
        ).all()
        print(f"[migrate] Found {len(paid_users)} gold/platinum users with no trial date — marking as paid")
        for u in paid_users:
            u.is_trial = False
            print(f"  → {u.email}: {u.plan_tier} — marked as paid (no trial)")

        db.commit()
        print("[migrate] Migration complete")

    except Exception as e:
        db.rollback()
        print(f"[migrate] ERROR: {e}")
        raise
    finally:
        db.close()


if __name__ == "__main__":
    run()
