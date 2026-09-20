"""Avis API client with retry logic, idempotency, and structured error handling.

Handles transient 5xx errors with exponential backoff. Write operations accept
an idempotency key to prevent duplicate side effects on retry.
"""
import os
import json
import logging

import requests
from dotenv import load_dotenv
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential_jitter,
    retry_if_exception,
)

load_dotenv()

AVIS_API_URL = os.environ["AVIS_API_URL"].rstrip("/")
AVIS_API_KEY = os.environ["AVIS_API_KEY"]

logger = logging.getLogger("avis_client")


# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------

class AvisAPIError(Exception):
    """A deterministic API error (4xx). Should NOT be retried."""

    def __init__(self, status_code: int, code: str, message: str, details: dict | None = None):
        self.status_code = status_code
        self.code = code
        self.message = message
        self.details = details or {}
        super().__init__(f"[{code}] {message}")


class AvisAPIUnavailable(Exception):
    """The API is unreachable after exhausting retries."""

    def __init__(self, message: str = "The Avis system is temporarily unavailable. Please try again in a moment."):
        super().__init__(message)


# ---------------------------------------------------------------------------
# Retry helpers
# ---------------------------------------------------------------------------

class _RetryableError(Exception):
    """Wrapper so tenacity knows to retry."""
    pass


def _is_retryable(exc: BaseException) -> bool:
    return isinstance(exc, _RetryableError)


_retry_policy = retry(
    retry=retry_if_exception(_is_retryable),
    stop=stop_after_attempt(3),
    wait=wait_exponential_jitter(initial=0.5, max=4, jitter=0.5),
    reraise=True,
)


# ---------------------------------------------------------------------------
# Base request helpers
# ---------------------------------------------------------------------------

def _handle_response(resp: requests.Response) -> dict:
    """Parse a response, raising AvisAPIError for 4xx or _RetryableError for 5xx."""
    if resp.status_code >= 500:
        logger.warning("Transient %d from API: %s", resp.status_code, resp.text[:200])
        raise _RetryableError(f"Server error {resp.status_code}")

    data = resp.json()

    if 400 <= resp.status_code < 500:
        err = data.get("error", {})
        raise AvisAPIError(
            status_code=resp.status_code,
            code=err.get("code", "UNKNOWN"),
            message=err.get("message", resp.text),
            details=err.get("details"),
        )

    return data


@_retry_policy
def _request(method: str, path: str, timeout: int = 10, **kwargs) -> dict:
    """Make an authenticated request to the Avis API with retry on transient errors."""
    headers = {"X-API-Key": AVIS_API_KEY}
    headers.update(kwargs.pop("headers", {}))

    try:
        resp = requests.request(
            method,
            f"{AVIS_API_URL}{path}",
            headers=headers,
            timeout=timeout,
            **kwargs,
        )
    except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
        logger.warning("Request to %s failed: %s", path, e)
        raise _RetryableError(str(e)) from e

    return _handle_response(resp)


def _write_request(method: str, path: str, body: dict,
                   idempotency_key: str | None = None, timeout: int = 20) -> dict:
    """Make a write request with JSON body and optional idempotency key."""
    headers = {"Content-Type": "application/json"}
    if idempotency_key:
        headers["Idempotency-Key"] = idempotency_key
    return _request(method, path, timeout=timeout, headers=headers, data=json.dumps(body))


# ---------------------------------------------------------------------------
# Endpoint functions
# ---------------------------------------------------------------------------

def get_reservation(reservation_id: str) -> dict:
    """Look up a reservation by ID."""
    try:
        return _request("GET", f"/reservations/{reservation_id}")
    except _RetryableError:
        raise AvisAPIUnavailable()


def check_availability(location: str, vehicle_type: str,
                       start_date: str, end_date: str) -> dict:
    """Check vehicle availability at a location for a date range."""
    try:
        return _request("GET", "/availability", params={
            "location": location,
            "vehicle_type": vehicle_type,
            "start_date": start_date,
            "end_date": end_date,
        })
    except _RetryableError:
        raise AvisAPIUnavailable()


def get_quote(reservation_id: str, change_type: str, new_return_datetime: str,
              new_return_location: str | None = None) -> dict:
    """Price a proposed change (no side effects)."""
    body = {"change_type": change_type, "new_return_datetime": new_return_datetime}
    if new_return_location:
        body["new_return_location"] = new_return_location
    try:
        return _write_request("POST", f"/reservations/{reservation_id}/quote", body)
    except _RetryableError:
        raise AvisAPIUnavailable()


def extend_reservation(reservation_id: str, new_return_datetime: str,
                       email: str, cvv: str, billing_zip: str,
                       idempotency_key: str) -> dict:
    """Extend a rental to a new return date/time."""
    body = {
        "new_return_datetime": new_return_datetime,
        "email": email,
        "payment": {"use_card_on_file": True, "cvv": cvv, "billing_zip": billing_zip},
    }
    try:
        return _write_request(
            "POST", f"/reservations/{reservation_id}/extend", body, idempotency_key
        )
    except _RetryableError:
        raise AvisAPIUnavailable()


def cancel_reservation(reservation_id: str, email: str,
                       reason: str = "", idempotency_key: str = "") -> dict:
    """Cancel a reservation."""
    body = {"email": email}
    if reason:
        body["reason"] = reason
    try:
        return _write_request(
            "POST", f"/reservations/{reservation_id}/cancel", body,
            idempotency_key or None
        )
    except _RetryableError:
        raise AvisAPIUnavailable()


def modify_reservation(reservation_id: str, email: str, cvv: str, billing_zip: str,
                       new_pickup_datetime: str | None = None,
                       new_return_datetime: str | None = None,
                       new_return_location: str | None = None,
                       idempotency_key: str = "") -> dict:
    """Modify a reservation (change time or return location)."""
    body: dict = {
        "email": email,
        "payment": {"use_card_on_file": True, "cvv": cvv, "billing_zip": billing_zip},
    }
    if new_pickup_datetime:
        body["new_pickup_datetime"] = new_pickup_datetime
    if new_return_datetime:
        body["new_return_datetime"] = new_return_datetime
    if new_return_location:
        body["new_return_location"] = new_return_location
    try:
        return _write_request(
            "POST", f"/reservations/{reservation_id}/modify", body,
            idempotency_key or None
        )
    except _RetryableError:
        raise AvisAPIUnavailable()


def upgrade_customer(customer_id: str, email: str,
                     idempotency_key: str = "") -> dict:
    """Upgrade a standard customer to Avis Preferred."""
    body = {"email": email}
    try:
        return _write_request(
            "POST", f"/customers/{customer_id}/upgrade", body,
            idempotency_key or None
        )
    except _RetryableError:
        raise AvisAPIUnavailable()


# ---------------------------------------------------------------------------
# CLI test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    reservation = get_reservation("AVS-29471835")
    print("Connected. Sample reservation:")
    print(f"  {reservation['customer_name']} — {reservation['vehicle']['description']}")
    print(f"  returns {reservation['dates']['current_return_datetime']} "
          f"at {reservation['return_location']['code']}")
