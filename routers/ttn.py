"""
routers/ttn.py

TTN-specific booking routes.
These will be removed or adapted when migration to Flight API is complete.

/ttn/book        - Probe TTN booking endpoints (discovery/debug)
/ttn/checkout-link - Full TTN booking flow: book + generate FAT redirect URL
"""

from typing import List, Optional

import requests
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from config import TTN_BASE_URL
from providers.ttn import _get_ttn_api_key, _get_ttn_auth_key, _ttn_headers

router = APIRouter()


class TTNBookRequest(BaseModel):
    session_id: str
    recommendation_id: str
    email: Optional[str] = None


class TTNPassenger(BaseModel):
    type: str           # ADT, CHD, INF
    firstname: str
    lastname: str
    birthday: str       # DD-MM-YYYY
    gender: str         # M, F
    citizenship: str    # 2-letter
    docnum: str
    doc_expire_date: str  # DD-MM-YYYY


class TTNCheckoutLinkRequest(BaseModel):
    providerSessionId: str
    providerRecommendationId: str
    real_email: str
    real_phone: str
    passengers: List[TTNPassenger]


@router.post("/ttn/book")
def ttn_book(payload: TTNBookRequest):
    """
    TTN booking probe.
    Phase 1 goal: discover the correct TTN booking/payment endpoint shape
    using session_id + recommendation_id.
    """
    api_key = _get_ttn_api_key()
    if not api_key:
        raise HTTPException(status_code=500, detail="TTN_API_KEY is not configured")

    print(f"[ttn] book.probe START session_id={payload.session_id} rec_id={payload.recommendation_id}")

    def _req(method: str, path: str, json_payload: Optional[dict] = None) -> dict:
        url = f"{TTN_BASE_URL}{path}"
        params = {"key": api_key}

        if method.upper() == "GET":
            params.update({
                "session_id": payload.session_id,
                "recommendation_id": payload.recommendation_id,
            })

        try:
            r = requests.request(
                method=method.upper(),
                url=url,
                params=params,
                json=json_payload if method.upper() != "GET" else None,
                headers=_ttn_headers(),
                timeout=30,
            )
        except Exception as e:
            return {"path": path, "method": method.upper(), "ok": False, "error": str(e)}

        body = (r.text or "")[:1200]
        out = {
            "path": path,
            "method": method.upper(),
            "status_code": r.status_code,
            "content_type": r.headers.get("content-type"),
            "body_preview": body,
        }

        if "application/json" in (out["content_type"] or ""):
            try:
                out["json"] = r.json()
            except Exception:
                pass

        return out

    candidates = [
        "/avia/book.json",
        "/avia/booking/create.json",
        "/avia/booking.json",
        "/avia/booking/start.json",
        "/avia/booking/confirm.json",
        "/avia/order/create.json",
        "/avia/order.json",
        "/avia/payment.json",
        "/avia/pay.json",
        "/avia/checkout.json",
        "/avia/ticketing.json",
        "/avia/issue.json",
    ]

    results = []
    for p in candidates:
        r_get = _req("GET", p)
        results.append(r_get)

        if r_get.get("status_code") and r_get["status_code"] != 404:
            print(f"[ttn] book.probe HIT method=GET path={p} status={r_get['status_code']}")
            return {"status": "probe", "hit": r_get, "all": results}

    return {"status": "probe", "hit": None, "all": results}


@router.post("/ttn/checkout-link")
def ttn_checkout_link(payload: TTNCheckoutLinkRequest):
    api_key = _get_ttn_api_key()
    if not api_key:
        raise HTTPException(status_code=500, detail="TTN_API_KEY is not configured")

    auth_key = _get_ttn_auth_key()
    if not auth_key:
        raise HTTPException(status_code=500, detail="TTN_AUTH_KEY is not configured")

    if not payload.passengers:
        raise HTTPException(status_code=400, detail="passengers is required")

    params = {
        "key": api_key,
        "session_id": payload.providerSessionId,
        "recommendation_id": payload.providerRecommendationId,
        "auth_key": auth_key,
        "real_email": payload.real_email,
        "real_phone": payload.real_phone,
    }

    for i, p in enumerate(payload.passengers):
        params[f"passengers[{i}][type]"] = p.type
        params[f"passengers[{i}][firstname]"] = p.firstname
        params[f"passengers[{i}][lastname]"] = p.lastname
        params[f"passengers[{i}][birthday]"] = p.birthday
        params[f"passengers[{i}][gender]"] = p.gender
        params[f"passengers[{i}][citizenship]"] = p.citizenship
        params[f"passengers[{i}][docnum]"] = p.docnum
        params[f"passengers[{i}][doc_expire_date]"] = p.doc_expire_date

    # 1) Create booking
    book_url = f"{TTN_BASE_URL}/avia/book.json"
    r = requests.get(book_url, params=params, headers=_ttn_headers(), timeout=30)

    ct = (r.headers.get("content-type") or "").lower()
    body_preview = (r.text or "")[:1200]

    if r.status_code != 200:
        raise HTTPException(
            status_code=502,
            detail={"stage": "book", "status_code": r.status_code, "content_type": ct, "body_preview": body_preview},
        )

    booking_id = None
    locator = None
    ttl = None

    book_json = None
    if "application/json" in ct:
        try:
            book_json = r.json()
        except Exception:
            book_json = None

    if isinstance(book_json, dict):
        resp = book_json.get("response") or {}
        booking = resp.get("booking") or {}
        locator = booking.get("locator")
        booking_id = booking.get("booking-id") or booking.get("booking_id")
        ttl = booking.get("ticketing-time-limit") or booking.get("ticketing_time_limit")

    if (not locator) and ("xml" in ct or (r.text or "").lstrip().startswith("<")):
        try:
            import xml.etree.ElementTree as ET
            root = ET.fromstring(r.text)
            b = root.find(".//booking")
            if b is not None:
                locator = b.get("locator") or locator
                booking_id = b.get("booking-id") or booking_id
                ttl = b.get("ticketing-time-limit") or ttl
        except Exception:
            pass

    if not locator:
        raise HTTPException(
            status_code=502,
            detail={"stage": "book_parse", "message": "Could not extract locator from booking response", "body_preview": body_preview},
        )

    # 2) Generate FAT redirect
    redirect_candidates = [
        "/payment/redirect.json",
        "/payment/redirect.xml",
        "/payment/redirect",
        "/avia/payment/redirect.json",
        "/avia/payment/redirect.xml",
    ]

    redirect_result = None
    for path in redirect_candidates:
        url = f"{TTN_BASE_URL}{path}"
        rp = {"key": api_key, "auth_key": auth_key, "service": "avia", "order_id": locator}
        rr = requests.get(url, params=rp, headers=_ttn_headers(), timeout=30)
        rct = (rr.headers.get("content-type") or "").lower()
        rbody = (rr.text or "")[:1200]

        if rr.status_code == 404:
            continue

        if "application/json" in rct:
            try:
                j = rr.json()
            except Exception:
                j = None
            if isinstance(j, dict):
                resp = j.get("response") or {}
                link = resp.get("link") or j.get("link") or {}
                checkout_url = link.get("url") or resp.get("url") or j.get("url")
                if checkout_url:
                    redirect_result = {"checkout_url": checkout_url, "raw": j, "path": path}
                    break

        if redirect_result is None and ("xml" in rct or (rr.text or "").lstrip().startswith("<")):
            try:
                import xml.etree.ElementTree as ET
                root = ET.fromstring(rr.text)
                link = root.find(".//link")
                if link is not None and link.get("url"):
                    redirect_result = {"checkout_url": link.get("url"), "raw": rbody, "path": path}
                    break
            except Exception:
                pass

        redirect_result = {"error": "no_checkout_url_in_response", "path": path, "content_type": rct, "body_preview": rbody}

    if not redirect_result or not redirect_result.get("checkout_url"):
        raise HTTPException(
            status_code=502,
            detail={"stage": "redirect", "message": "Could not generate FAT redirect URL", "locator": locator, "probe": redirect_result},
        )

    return {
        "status": "ok",
        "locator": locator,
        "booking_id": booking_id,
        "ticketing_time_limit": ttl,
        "checkout_url": redirect_result["checkout_url"],
        "redirect_path": redirect_result.get("path"),
    }
