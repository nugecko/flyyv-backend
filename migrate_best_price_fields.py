"""
migrate_best_price_fields.py
-----------------------------
Adds best_price_departure_date and best_price_booking_urls to the alerts table.

Run once after deploying:
  dokku run flyyv python migrate_best_price_fields.py

Safe to run multiple times — uses ADD COLUMN IF NOT EXISTS.
"""

from sqlalchemy import text
from db import engine


def run():
    print("[migrate] Adding best price fields to alerts table...")
    with engine.connect() as conn:
        conn.execute(text("""
            ALTER TABLE alerts
            ADD COLUMN IF NOT EXISTS best_price_departure_date DATE NULL;
        """))
        conn.execute(text("""
            ALTER TABLE alerts
            ADD COLUMN IF NOT EXISTS best_price_booking_urls JSONB NULL;
        """))
        conn.commit()
    print("[migrate] Done — columns added (or already existed)")


if __name__ == "__main__":
    run()
