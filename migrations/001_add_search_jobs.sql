-- migrations/001_add_search_jobs.sql
--
-- Replaces in-memory JOBS / JOB_RESULTS with Postgres-backed tables.
-- Run once on the Dokku Postgres instance before deploying the new code.
--
-- SQLAlchemy will CREATE these automatically via Base.metadata.create_all()
-- if they don't exist yet, so this script is only needed if you prefer
-- to manage schema changes manually.

-- Search jobs: one row per async scan job
CREATE TABLE IF NOT EXISTS search_jobs (
    id              VARCHAR(255) PRIMARY KEY,
    status          VARCHAR(50)  NOT NULL DEFAULT 'pending',
    params_json     JSONB        NOT NULL,
    total_pairs     INTEGER      NOT NULL DEFAULT 0,
    processed_pairs INTEGER      NOT NULL DEFAULT 0,
    error           TEXT,
    created_at      TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_search_jobs_status    ON search_jobs (status);
CREATE INDEX IF NOT EXISTS idx_search_jobs_created   ON search_jobs (created_at);

-- Search results: one row per FlightOption returned by a job
-- NOTE: For high-throughput, consider partitioning by created_at.
CREATE TABLE IF NOT EXISTS search_results (
    id          UUID         PRIMARY KEY DEFAULT gen_random_uuid(),
    job_id      VARCHAR(255) NOT NULL REFERENCES search_jobs(id) ON DELETE CASCADE,
    offer_json  JSONB        NOT NULL,
    created_at  TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_search_results_job    ON search_results (job_id);

-- Auto-cleanup: delete jobs older than 24 hours (optional, run as cron)
-- DELETE FROM search_jobs WHERE created_at < NOW() - INTERVAL '24 hours';
