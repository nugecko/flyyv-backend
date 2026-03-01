"""routers/search.py - Search routes: start, status, results."""

import threading
from datetime import datetime
from typing import Optional
from uuid import uuid4

from fastapi import APIRouter, BackgroundTasks

from config import get_config_int, get_config_str
from providers.factory import run_provider_scan
from providers.ttn import _get_ttn_api_key
from schemas.search import (
    FlightOption,
    JobStatus,
    SearchJob,
    SearchParams,
    SearchResultsResponse,
    SearchStatusResponse,
)
from services.search_service import (
    JOBS,
    JOB_RESULTS,
    _GLOBAL_SEARCH_SEM,
    _USER_GUARD_LOCK,
    _USER_INFLIGHT,
    apply_global_airline_cap,
    deduplicate_by_outbound,
    begin_user_inflight,
    end_user_inflight,
    estimate_date_pairs,
    get_search_hard_cap,
    peek_user_inflight,
    run_search_job_guarded,
    user_key_from_params,
    _hard_runtime_cap,
)

router = APIRouter()

ESTIMATED_SECONDS_PER_PAIR = 6.0


@router.post("/search-business")
def search_business(params: SearchParams, background_tasks: BackgroundTasks):
    request_id = str(uuid4())

    if not _get_ttn_api_key():
        return {"status": "error", "source": "ttn_not_configured", "options": []}

    print(
        f"[search_business] search_mode={getattr(params,'search_mode',None)} "
        f"earliestDeparture={getattr(params,'earliestDeparture',None)} "
        f"latestDeparture={getattr(params,'latestDeparture',None)} "
        f"nights={getattr(params,'nights',None)} "
        f"minStayDays={getattr(params,'minStayDays',None)} "
        f"maxStayDays={getattr(params,'maxStayDays',None)}"
    )

    max_passengers = get_config_int("MAX_PASSENGERS", 4)
    if params.passengers > max_passengers:
        params.passengers = max_passengers

    default_cabin = get_config_str("DEFAULT_CABIN", "BUSINESS") or "BUSINESS"
    if not params.cabin:
        params.cabin = default_cabin

    user_key = user_key_from_params(params)
    estimated_pairs = estimate_date_pairs(params)

    inflight_job_id = None
    inflight_age = None
    if user_key:
        inflight_job_id, inflight_age = peek_user_inflight(user_key)

        print(
            f"[trace] request_id={request_id} user_key={repr(user_key)} "
            f"estimated_pairs={estimated_pairs} inflight_job_id={inflight_job_id} "
            f"inflight_age_s={None if inflight_age is None else int(inflight_age)} "
            f"origin={getattr(params,'origin',None)} dest={getattr(params,'destination',None)} "
            f"search_mode={getattr(params,'search_mode',None)}"
        )
        print(f"[guardrail] user_external_id_raw={repr(getattr(params, 'user_external_id', None))} user_key={repr(user_key)}")

    # Anonymous users: only allow single-pair searches
    if not user_key and estimated_pairs > 1:
        print(
            f"[guardrail] missing_user_id "
            f"origin={params.origin} dest={params.destination} "
            f"earliest={params.earliestDeparture} latest={params.latestDeparture}"
        )
        return {
            "status": "error",
            "source": "missing_user_id",
            "message": "Missing user identity for multi-date searches. Please sign in and retry.",
        }

    # Per-user single-flight guard (REUSE mode)
    new_job_id = None

    if user_key:
        inflight_job_id, inflight_age = peek_user_inflight(user_key)

        if inflight_job_id:
            j = JOBS.get(inflight_job_id)
            if j and j.status in (JobStatus.PENDING, JobStatus.RUNNING):
                print(
                    f"[guardrail] reuse_inflight user_key={user_key} job_id={inflight_job_id} "
                    f"age_s={None if inflight_age is None else int(inflight_age)}"
                )
                return {
                    "status": "ok",
                    "mode": "async",
                    "jobId": inflight_job_id,
                    "message": "Search already running, reusing existing job",
                }

        if estimated_pairs > 1:
            new_job_id = str(uuid4())

            begin_ok = begin_user_inflight(user_key, job_id=new_job_id)
            if not begin_ok:
                blocker_job_id, blocker_age = peek_user_inflight(user_key)

                print(
                    f"[trace] request_id={request_id} begin_ok=False user_key={repr(user_key)} "
                    f"blocker_job_id={blocker_job_id} "
                    f"blocker_age_s={None if blocker_age is None else int(blocker_age)}"
                )

                if blocker_job_id:
                    print(f"[guardrail] returning_existing_job user_key={user_key} job_id={blocker_job_id}")
                    return {
                        "status": "ok",
                        "mode": "async",
                        "jobId": blocker_job_id,
                        "message": "Search already running, returning existing job.",
                    }

                print(f"[guardrail] search_in_progress user_key={user_key}")
                return {
                    "status": "error",
                    "source": "search_in_progress",
                    "message": "A search is already running for this user, please wait for it to finish.",
                }

            print(f"[trace] request_id={request_id} begin_ok=True user_key={repr(user_key)} job_id={new_job_id}")

    # Global concurrency guard
    acquired = _GLOBAL_SEARCH_SEM.acquire(blocking=False)

    if not acquired:
        if user_key:
            end_user_inflight(user_key)
        print(f"[guardrail] server_busy user_key={user_key}")
        return {
            "status": "error",
            "source": "server_busy",
            "message": "Server is busy running other searches, please retry in a moment.",
        }

    try:
        # Sync path (single pair)
        if estimated_pairs <= 1:
            with _hard_runtime_cap(get_search_hard_cap(), job_id=None):
                options = run_provider_scan(params)
                options = deduplicate_by_outbound(options)
                options = apply_global_airline_cap(options, max_share=0.33)
                return {
                    "status": "ok",
                    "mode": "sync",
                    "source": "ttn",
                    "options": [o.dict() for o in options],
                }

        # Async path (multi pair)
        job_id = new_job_id or str(uuid4())
        job = SearchJob(
            id=job_id,
            status=JobStatus.PENDING,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            params=params,
            total_pairs=0,
            processed_pairs=0,
        )
        JOBS[job_id] = job
        JOB_RESULTS[job_id] = []

        if user_key:
            with _USER_GUARD_LOCK:
                if user_key in _USER_INFLIGHT:
                    _USER_INFLIGHT[user_key]["job_id"] = job_id

                t = threading.Thread(
                    target=run_search_job_guarded,
                    args=(job_id, user_key),
                    daemon=True,
                    name=f"search-job-{job_id[:8]}",
                )
                t.start()
        else:
            _GLOBAL_SEARCH_SEM.release()
            return {
                "status": "error",
                "source": "missing_user_id",
                "message": "Missing user identity for multi-date searches. Please sign in and retry.",
            }

        return {"status": "ok", "mode": "async", "jobId": job_id, "message": "Search started"}

    except Exception as e:
        if user_key:
            end_user_inflight(user_key)
        _GLOBAL_SEARCH_SEM.release()
        raise e

    finally:
        if estimated_pairs <= 1:
            if user_key:
                end_user_inflight(user_key)
            _GLOBAL_SEARCH_SEM.release()


@router.get("/search-status/{job_id}", response_model=SearchStatusResponse)
def get_search_status(job_id: str, preview_limit: int = 20):
    job = JOBS.get(job_id)

    if not job:
        return SearchStatusResponse(
            jobId=job_id,
            status=JobStatus.CANCELLED,
            processedPairs=0,
            totalPairs=0,
            progress=0.0,
            error="Job not found in memory, the server likely restarted. Please start a new search.",
            previewCount=0,
            previewOptions=[],
            elapsedSeconds=None,
            estimatedTotalSeconds=None,
            estimatedProgressPct=None,
        )

    options = JOB_RESULTS.get(job_id, [])
    preview = options[:preview_limit] if preview_limit > 0 else []

    total_pairs = job.total_pairs or 0
    processed_pairs = job.processed_pairs or 0
    print(f"[status] job_id={job_id[:8]} status={job.status} processed={processed_pairs}/{total_pairs}")
    progress = float(processed_pairs) / float(total_pairs) if total_pairs > 0 else 0.0

    elapsed_seconds = None
    estimated_total_seconds = None
    estimated_progress_pct = None

    if job.status == JobStatus.RUNNING and total_pairs > 0:
        now = datetime.utcnow()
        elapsed_seconds = (now - job.created_at).total_seconds()
        estimated_total_seconds = total_pairs * ESTIMATED_SECONDS_PER_PAIR

        time_based_pct = min(95.0, (elapsed_seconds / estimated_total_seconds) * 100) if estimated_total_seconds > 0 else 0.0
        actual_pct = progress * 100
        estimated_progress_pct = max(time_based_pct, actual_pct)

        if processed_pairs > 0:
            actual_seconds_per_pair = elapsed_seconds / processed_pairs
            estimated_total_seconds = total_pairs * actual_seconds_per_pair

    elif job.status == JobStatus.COMPLETED:
        elapsed_seconds = (job.updated_at - job.created_at).total_seconds() if job.updated_at else None
        estimated_total_seconds = elapsed_seconds
        estimated_progress_pct = 100.0

    return SearchStatusResponse(
        jobId=job.id,
        status=job.status,
        processedPairs=processed_pairs,
        totalPairs=total_pairs,
        progress=progress,
        error=job.error,
        previewCount=len(preview),
        previewOptions=preview,
        elapsedSeconds=round(elapsed_seconds, 1) if elapsed_seconds is not None else None,
        estimatedTotalSeconds=round(estimated_total_seconds, 1) if estimated_total_seconds is not None else None,
        estimatedProgressPct=round(estimated_progress_pct, 1) if estimated_progress_pct is not None else None,
    )


@router.get("/search-results/{job_id}", response_model=SearchResultsResponse)
def get_search_results(job_id: str, offset: int = 0, limit: int = 50):
    job = JOBS.get(job_id)
    if not job:
        return SearchResultsResponse(
            jobId=job_id,
            status=JobStatus.PENDING,
            totalResults=0,
            offset=0,
            limit=limit,
            options=[],
        )
    options = JOB_RESULTS.get(job_id, [])
    offset = max(0, offset)
    limit = max(1, min(limit, 600))
    end = min(offset + limit, len(options))
    slice_ = options[offset:end]
    return SearchResultsResponse(
        jobId=job.id,
        status=job.status,
        totalResults=len(options),
        offset=offset,
        limit=limit,
        options=slice_,
    )