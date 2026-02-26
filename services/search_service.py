"""
services/search_service.py

All search execution logic:
- In-memory job store (JOBS, JOB_RESULTS)
- Per-user and global concurrency guardrails
- Date pair generation (including FlyyvFlex fixed-nights mode)
- Filtering and airline balancing
- Async job runner (process_date_pair_offers, run_search_job)

NOTE ON JOB PERSISTENCE:
JOBS and JOB_RESULTS are currently in-memory only. This means active jobs
are lost on container restart. A full Postgres migration is planned:
  - See migrations/001_add_search_jobs.sql
  - Migrate JOBS → search_jobs table
  - Migrate JOB_RESULTS → search_results table
For now, all job state lives here.
"""

import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from contextlib import contextmanager
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple
from uuid import uuid4

from config import (
    MAX_DATE_PAIRS_HARD,
    MAX_OFFERS_PER_PAIR_HARD,
    MAX_OFFERS_TOTAL_HARD,
    get_config_int,
    get_config_str,
)
from providers.factory import run_provider_scan
from schemas.search import FlightOption, JobStatus, SearchJob, SearchParams


# =====================================================================
# SECTION: IN-MEMORY STATE
# These dicts are module-level. They are shared across all threads.
# They vanish on container restart. See migration note above.
# =====================================================================

JOBS: Dict[str, SearchJob] = {}
JOB_RESULTS: Dict[str, List[FlightOption]] = {}

# Admin credits wallet (in-memory only, not persisted)
USER_WALLETS: Dict[str, int] = {}


# =====================================================================
# SECTION: GUARDRAILS
# =====================================================================

_MAX_CONCURRENT_SEARCHES = get_config_int("MAX_CONCURRENT_SEARCHES", 2)
_SEARCH_HARD_CAP_SECONDS = get_config_int("SEARCH_HARD_CAP_SECONDS", 70)

_GLOBAL_SEARCH_SEM = threading.Semaphore(_MAX_CONCURRENT_SEARCHES)
_USER_GUARD_LOCK = threading.Lock()
_USER_INFLIGHT: Dict[str, Dict[str, Any]] = {}  # user_key -> {"job_id": str|None, "started_at": float}
_HARD_CAP_HIT: Set[str] = set()  # set of job_ids that hit the hard cap


def get_search_hard_cap() -> int:
    return get_config_int("SEARCH_HARD_CAP_SECONDS", _SEARCH_HARD_CAP_SECONDS)


def user_key_from_params(params: SearchParams) -> Optional[str]:
    """
    IMPORTANT: Per-user single-flight guard only works if this is truly stable.
    Do NOT fall back to origin/destination for async scans.
    """
    for k in ("user_external_id", "userExternalId", "user_email", "userEmail"):
        v = getattr(params, k, None)
        if v:
            return str(v).strip().lower()
    return None


@contextmanager
def _hard_runtime_cap(seconds: int, job_id: Optional[str] = None):
    """
    Hard cap: if work hangs, mark the job capped and let the loop cancel safely.
    Does NOT kill the web process.
    """
    if not seconds or seconds <= 0:
        yield
        return

    def _flag():
        if job_id:
            try:
                _HARD_CAP_HIT.add(job_id)
            except Exception:
                pass
        print(f"[guardrail] HARD CAP HIT after {seconds}s job_id={job_id}")

    t = threading.Timer(seconds, _flag)
    t.daemon = True
    t.start()
    try:
        yield
    finally:
        t.cancel()


def begin_user_inflight(user_key: str, job_id: Optional[str]) -> bool:
    now = time.monotonic()
    with _USER_GUARD_LOCK:
        rec = _USER_INFLIGHT.get(user_key)
        if rec and (now - rec.get("started_at", now)) > (get_search_hard_cap() + 30):
            print(f"[guardrail] stale_inflight_cleared user_key={user_key}")
            _USER_INFLIGHT.pop(user_key, None)

        if user_key in _USER_INFLIGHT:
            return False

        _USER_INFLIGHT[user_key] = {"job_id": job_id, "started_at": now}
        return True


def end_user_inflight(user_key: str):
    with _USER_GUARD_LOCK:
        _USER_INFLIGHT.pop(user_key, None)


def peek_user_inflight(user_key: str) -> Tuple[Optional[str], Optional[float]]:
    """Return (job_id, age_seconds) for current inflight record, or (None, None)."""
    now = time.monotonic()
    with _USER_GUARD_LOCK:
        rec = _USER_INFLIGHT.get(user_key) or {}
        job_id = rec.get("job_id")
        started_at = rec.get("started_at")
    age = (now - started_at) if started_at else None
    return job_id, age


# =====================================================================
# SECTION: EFFECTIVE CAPS
# =====================================================================

def effective_caps(params: SearchParams) -> Tuple[int, int, int]:
    config_max_pairs = get_config_int("MAX_DATE_PAIRS", 60)
    max_pairs = max(1, min(config_max_pairs, MAX_DATE_PAIRS_HARD))

    requested_per_pair = max(1, params.maxOffersPerPair)
    requested_total = max(1, params.maxOffersTotal)

    config_max_offers_pair = get_config_int("MAX_OFFERS_PER_PAIR", 80)
    config_max_offers_total = get_config_int("MAX_OFFERS_TOTAL", 4000)

    max_offers_pair = max(1, min(requested_per_pair, config_max_offers_pair, MAX_OFFERS_PER_PAIR_HARD))
    max_offers_total = max(1, min(requested_total, config_max_offers_total, MAX_OFFERS_TOTAL_HARD))

    return max_pairs, max_offers_pair, max_offers_total


# =====================================================================
# SECTION: DATE PAIR GENERATION
# =====================================================================

def generate_date_pairs(params: Any, max_pairs: int = 60) -> List[Tuple[date, date]]:
    """
    Build (departure_date, return_date) pairs.

    Rules:
      - Single searches: allow minStayDays..maxStayDays
      - FlyyvFlex (search_mode == "flexible"): trip length is fixed
        Use a single "nights" value:
          1) params.nights if present
          2) else params.minStayDays
          3) else 0
        Generate one return per departure: return = departure + nights
        Skip if return > latestDeparture
    """
    earliest = getattr(params, "earliestDeparture", None)
    latest = getattr(params, "latestDeparture", None)

    if not earliest or not latest or earliest > latest:
        return []

    print(
        f"[pairs] search_mode={getattr(params,'search_mode',None)} earliest={earliest} latest={latest} "
        f"nights={getattr(params,'nights',None)} minStayDays={getattr(params,'minStayDays',None)} "
        f"maxStayDays={getattr(params,'maxStayDays',None)} maxDatePairs={getattr(params,'maxDatePairs',None)}"
    )

    param_cap = getattr(params, "maxDatePairs", None)
    if isinstance(param_cap, int) and param_cap > 0:
        max_pairs = min(max_pairs, param_cap)

    pairs: List[Tuple[date, date]] = []
    search_mode = getattr(params, "search_mode", None)

    # FlyyvFlex: fixed nights
    if (
        (search_mode == "flexible")
        or (getattr(params, "nights", None) is not None)
        or (
            (getattr(params, "minStayDays", None) is not None)
            and (getattr(params, "maxStayDays", None) is not None)
            and int(getattr(params, "minStayDays") or 0) > 0
            and int(getattr(params, "minStayDays") or 0) == int(getattr(params, "maxStayDays") or 0)
        )
    ):
        nights = getattr(params, "nights", None)
        if nights is None:
            nights = getattr(params, "minStayDays", None)
        nights = int(nights or 0)
        if nights < 0:
            nights = 0

        last_dep = latest - timedelta(days=nights)

        dep = earliest
        while dep <= last_dep and len(pairs) < max_pairs:
            ret = dep + timedelta(days=nights)
            pairs.append((dep, ret))
            dep = dep + timedelta(days=1)

        print(
            f"[pairs_flex] nights={nights} earliest={earliest} latest={latest} "
            f"last_dep={last_dep} len={len(pairs)} "
            f"first_pair={pairs[0] if pairs else None} "
            f"last_pair={pairs[-1] if pairs else None}"
        )
        return pairs

    # Default: range of stay lengths
    min_stay = max(0, int(getattr(params, "minStayDays", 0) or 0))
    max_stay = max(min_stay, int(getattr(params, "maxStayDays", min_stay) or min_stay))

    dep = earliest
    while dep <= latest and len(pairs) < max_pairs:
        for stay in range(min_stay, max_stay + 1):
            ret = dep + timedelta(days=stay)
            pairs.append((dep, ret))
            if len(pairs) >= max_pairs:
                break
        dep = dep + timedelta(days=1)

    return pairs


def estimate_date_pairs(params: SearchParams) -> int:
    max_pairs, _, _ = effective_caps(params)
    pairs = generate_date_pairs(params, max_pairs=max_pairs)
    return len(pairs)


# =====================================================================
# SECTION: FILTERING AND BALANCING
# =====================================================================

def apply_filters(options: List[FlightOption], params: SearchParams) -> List[FlightOption]:
    filtered = list(options)

    if filtered:
        stops_dist = Counter(o.stops for o in filtered)
        print(
            f"[search] apply_filters: input={len(filtered)} stops_dist={dict(stops_dist)} "
            f"maxPrice={params.maxPrice} stopsFilter={params.stopsFilter}"
        )
    else:
        print("[search] apply_filters: input=0")

    if params.maxPrice is not None and params.maxPrice > 0:
        before = len(filtered)
        filtered = [o for o in filtered if o.price <= params.maxPrice]
        print(f"[search] apply_filters: maxPrice kept={len(filtered)}/{before}")

    if params.stopsFilter:
        before = len(filtered)
        allowed = set(params.stopsFilter)
        if 3 in allowed:
            filtered = [o for o in filtered if (o.stops in allowed or o.stops >= 3)]
        else:
            filtered = [o for o in filtered if o.stops in allowed]
        print(f"[search] apply_filters: stopsFilter kept={len(filtered)}/{before} allowed={sorted(list(allowed))}")

    filtered.sort(key=lambda o: (o.stops, o.price))
    return filtered


def balance_airlines(
    options: List[FlightOption],
    max_total: Optional[int] = None,
) -> List[FlightOption]:
    if not options:
        return []

    sorted_by_price = sorted(options, key=lambda x: x.price)

    if max_total is None or max_total <= 0:
        max_total = len(sorted_by_price)

    actual_total = min(max_total, len(sorted_by_price))

    max_share_percent = get_config_int("MAX_AIRLINE_SHARE_PERCENT", 40)
    if max_share_percent <= 0 or max_share_percent > 100:
        max_share_percent = 40

    airline_counts: Dict[str, int] = defaultdict(int)
    result: List[FlightOption] = []

    airline_buckets: Dict[str, List[FlightOption]] = defaultdict(list)
    for opt in sorted_by_price:
        key = opt.airlineCode or opt.airline
        airline_buckets[key].append(opt)

    unique_airlines = list(airline_buckets.keys())
    num_airlines = max(1, len(unique_airlines))

    base_cap = max(1, (max_share_percent * actual_total) // 100)
    per_airline_cap = max(base_cap, actual_total // num_airlines if num_airlines else base_cap)

    seen_ids = set()
    for airline_key, bucket in airline_buckets.items():
        if len(result) >= actual_total:
            break
        cheapest_opt = bucket[0]
        if cheapest_opt is None:
            continue
        airline_counts[airline_key] += 1
        result.append(cheapest_opt)
        seen_ids.add(id(cheapest_opt))

    for opt in sorted_by_price:
        if len(result) >= actual_total:
            break
        if id(opt) in seen_ids:
            continue
        key = opt.airlineCode or opt.airline
        if airline_counts[key] >= per_airline_cap:
            continue
        airline_counts[key] += 1
        result.append(opt)
        seen_ids.add(id(opt))

    result.sort(key=lambda x: x.price)
    return result


def apply_global_airline_cap(
    options: List[FlightOption],
    max_share: float = 0.5,
) -> List[FlightOption]:
    if not options:
        print("[search] apply_global_airline_cap: no options, skipping")
        return options

    total = len(options)
    max_per_airline = max(1, int(total * max_share))
    counts: Counter = Counter()
    capped: List[FlightOption] = []

    for opt in options:
        airline = opt.airlineCode or opt.airline or "UNKNOWN"
        if counts[airline] >= max_per_airline:
            continue
        capped.append(opt)
        counts[airline] += 1

    print(
        f"[search] apply_global_airline_cap: input={total}, "
        f"output={len(capped)}, max_per_airline={max_per_airline}, "
        f"airline_counts={dict(counts)}"
    )
    return capped


# =====================================================================
# SECTION: ASYNC DATE-PAIR WORKER
# =====================================================================

def process_date_pair_offers(
    params: SearchParams,
    dep: date,
    ret: date,
    max_offers_pair: int,
) -> List[FlightOption]:
    """
    Fetch offers for exactly one (dep, ret) pair via the active provider.
    Called by the async job runner in a thread pool.
    """
    per_pair_limit = int(max_offers_pair) if max_offers_pair else 20
    per_pair_limit = max(1, min(per_pair_limit, 50))

    try:
        scan_params = SearchParams(**params.model_dump())
        scan_params.earliestDeparture = dep

        opts = run_provider_scan(scan_params) or []

        dep_iso = dep.isoformat()
        ret_iso = ret.isoformat()

        for o in opts:
            try:
                o.departureDate = dep_iso
                o.returnDate = ret_iso
            except Exception:
                pass

        opts = opts[:per_pair_limit]
        print(f"[pair_worker] dep={dep_iso} ret={ret_iso} mapped={len(opts)}")
        return opts

    except Exception as e:
        from fastapi import HTTPException
        if isinstance(e, HTTPException):
            print(f"[pair_worker] HTTPException dep={dep} ret={ret}: {e.detail}")
        else:
            print(f"[pair_worker] error dep={dep} ret={ret}: {e}")
        return []


# =====================================================================
# SECTION: LEGACY SINGLE-PAIR JOB RUNNER
# Used by the older run_search_job path (sequential, no batching).
# Kept for backward compatibility. The new async path uses the batch runner.
# =====================================================================

def run_search_job(job_id: str) -> None:
    job = JOBS.get(job_id)
    if not job:
        print(f"[JOB {job_id}] missing job, aborting")
        return

    try:
        job.status = JobStatus.RUNNING

        max_pairs, max_offers_pair, max_offers_total = effective_caps(job.params)
        pairs = generate_date_pairs(job.params, max_pairs=max_pairs)
        job.total_pairs = len(pairs)
        job.processed_pairs = 0

        all_options: List[FlightOption] = []
        JOB_RESULTS[job_id] = []

        PARALLEL_WORKERS = get_config_int("PARALLEL_WORKERS", 6)
        BATCH_SIZE = PARALLEL_WORKERS

        import time as _time

        with ThreadPoolExecutor(max_workers=PARALLEL_WORKERS) as executor:
            pending = set()
            pair_iter = iter(pairs)
            done_all = False

            while True:
                if job_id in _HARD_CAP_HIT:
                    print(f"[JOB {job_id}] hard cap hit, stopping batch loop")
                    for f in pending:
                        f.cancel()
                    break

                # Fill the batch
                while len(pending) < BATCH_SIZE and not done_all:
                    try:
                        dep, ret = next(pair_iter)
                        fut = executor.submit(process_date_pair_offers, job.params, dep, ret, max_offers_pair)
                        pending.add(fut)
                    except StopIteration:
                        done_all = True
                        break

                if not pending:
                    break

                # Wait for any future to complete
                done, pending = wait(pending, timeout=1, return_when=FIRST_COMPLETED)

                for fut in done:
                    try:
                        opts = fut.result()
                        if opts:
                            all_options.extend(opts)
                            JOB_RESULTS[job_id] = all_options
                    except Exception as e:
                        print(f"[JOB {job_id}] future failed: {e}")

                    job.processed_pairs += 1
                    job.updated_at = datetime.utcnow()

                    if len(all_options) >= max_offers_total:
                        print(f"[JOB {job_id}] max_offers_total reached, stopping")
                        done_all = True
                        for f in pending:
                            f.cancel()
                        pending = set()
                        break

        # Final global cap and balance
        if all_options:
            all_options = apply_global_airline_cap(all_options, max_share=0.5)
            JOB_RESULTS[job_id] = all_options

        job.status = JobStatus.COMPLETED
        job.updated_at = datetime.utcnow()
        print(f"[JOB {job_id}] done processed={job.processed_pairs}/{job.total_pairs} options={len(all_options)}")

    except Exception as e:
        job.status = JobStatus.FAILED
        job.error = f"job_crash: {type(e).__name__}: {e}"
        job.updated_at = datetime.utcnow()
        print(f"[JOB {job_id}] FAILED {job.error}")


# =====================================================================
# SECTION: GUARDED RUNNER (CALLED BY BACKGROUND THREAD)
# =====================================================================

def run_search_job_guarded(job_id: str, user_key: str) -> None:
    try:
        with _hard_runtime_cap(get_search_hard_cap(), job_id=job_id):
            run_search_job(job_id)
    except Exception as e:
        try:
            job = JOBS.get(job_id)
            if job:
                job.status = JobStatus.FAILED
                job.error = f"job_crash: {type(e).__name__}: {e}"
                job.updated_at = datetime.utcnow()
        except Exception as _e:
            print(f"[JOB {job_id}] FAILED status update failed: {_e}")
        raise
    finally:
        try:
            job = JOBS.get(job_id)
            st = getattr(job, "status", None)
            print(f"[JOB {job_id}] FINALLY reached status={st} user_key={user_key}")
        except Exception as _e:
            print(f"[JOB {job_id}] FINALLY reached log_failed: {_e}")

        end_user_inflight(user_key)
        _GLOBAL_SEARCH_SEM.release()
