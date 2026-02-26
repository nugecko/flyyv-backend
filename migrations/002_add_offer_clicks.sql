-- migrations/002_add_offer_clicks.sql
--
-- Click tracking table for offer analytics.
-- Logs every outbound booking click with full offer context.
-- All rows visible in Directus for admin analytics.

CREATE TABLE IF NOT EXISTS offer_clicks (
    id                  UUID         PRIMARY KEY DEFAULT gen_random_uuid(),
    clicked_at          TIMESTAMPTZ  NOT NULL DEFAULT NOW(),

    -- Who clicked
    user_external_id    VARCHAR(255),

    -- What was clicked
    offer_id            VARCHAR(512),
    airline_code        VARCHAR(10),
    price               NUMERIC(10, 2),
    currency            VARCHAR(10),
    origin              VARCHAR(10),
    destination         VARCHAR(10),
    departure_date      DATE,
    return_date         DATE,
    cabin               VARCHAR(50),
    stops               INTEGER,

    -- Context
    source              VARCHAR(50),   -- "search" | "alert_email" | "preview"
    job_id              VARCHAR(255),
    alert_id            VARCHAR(255),

    -- Destination
    destination_url     TEXT,
    provider            VARCHAR(50)   NOT NULL DEFAULT 'ttn',
    redirect_followed   BOOLEAN       NOT NULL DEFAULT TRUE
);

-- Indexes for common analytics queries
CREATE INDEX IF NOT EXISTS idx_offer_clicks_user        ON offer_clicks (user_external_id);
CREATE INDEX IF NOT EXISTS idx_offer_clicks_clicked_at  ON offer_clicks (clicked_at);
CREATE INDEX IF NOT EXISTS idx_offer_clicks_airline     ON offer_clicks (airline_code);
CREATE INDEX IF NOT EXISTS idx_offer_clicks_route       ON offer_clicks (origin, destination);
CREATE INDEX IF NOT EXISTS idx_offer_clicks_source      ON offer_clicks (source);
CREATE INDEX IF NOT EXISTS idx_offer_clicks_alert       ON offer_clicks (alert_id);
