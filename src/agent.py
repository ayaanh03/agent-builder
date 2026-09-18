"""Avis rental support agent — extend and cancel workflows with RAG.

Exports the `agent` object for use by the terminal UI. No main() here.
"""
import os
import uuid
import json

from dotenv import load_dotenv

load_dotenv()

from agents import Agent, Runner, function_tool  # noqa: E402

import time  # noqa: E402
import avis_client  # noqa: E402
from avis_client import AvisAPIError, AvisAPIUnavailable  # noqa: E402
import knowledge_base  # noqa: E402
from chat_logger import get_logger  # noqa: E402


def _log_tool(name: str, inputs: dict, output, success: bool = True, duration_ms: int | None = None):
    """Log a tool call to the active session logger if one exists."""
    logger = get_logger()
    if logger:
        logger.log_tool_call(name, inputs, output, success=success, duration_ms=duration_ms)


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

@function_tool
def lookup_reservation(reservation_id: str) -> str:
    """Look up an Avis reservation by its ID (e.g. 'AVS-29471835').
    Returns reservation details including customer, vehicle, dates, and status."""
    t0 = time.monotonic()
    try:
        data = avis_client.get_reservation(reservation_id)
        result = json.dumps(data, indent=2)
        _log_tool("lookup_reservation", {"reservation_id": reservation_id}, result,
                  duration_ms=int((time.monotonic() - t0) * 1000))
        return result
    except AvisAPIError as e:
        msg = f"Error looking up reservation: {e.message}"
        _log_tool("lookup_reservation", {"reservation_id": reservation_id}, msg,
                  success=False, duration_ms=int((time.monotonic() - t0) * 1000))
        return msg
    except AvisAPIUnavailable as e:
        _log_tool("lookup_reservation", {"reservation_id": reservation_id}, str(e),
                  success=False, duration_ms=int((time.monotonic() - t0) * 1000))
        return str(e)


@function_tool
def search_knowledge_base(query: str) -> str:
    """Search Avis help-center articles for policies, fees, and procedures.
    Use this to answer customer questions about rental policies."""
    t0 = time.monotonic()
    results = knowledge_base.search(query, top_k=3)
    if not results:
        out = "No relevant articles found for that query."
        _log_tool("search_knowledge_base", {"query": query}, out,
                  duration_ms=int((time.monotonic() - t0) * 1000))
        return out
    parts = []
    for r in results:
        parts.append(f"**{r['title']}** ({r['authority']})\n{r['body']}")
    out = "\n\n---\n\n".join(parts)
    _log_tool("search_knowledge_base", {"query": query},
              {"articles": [r["id"] for r in results]},
              duration_ms=int((time.monotonic() - t0) * 1000))
    return out


@function_tool
def get_extension_quote(reservation_id: str, new_return_datetime: str) -> str:
    """Get a price quote for extending a rental to a new return date/time.
    Date format: YYYY-MM-DDTHH:MM:SS with timezone offset (e.g. 2027-06-17T14:00:00-07:00).
    This is a read-only operation — it does not commit the change."""
    t0 = time.monotonic()
    inputs = {"reservation_id": reservation_id, "new_return_datetime": new_return_datetime}
    try:
        data = avis_client.get_quote(reservation_id, "extend", new_return_datetime)
        result = json.dumps(data, indent=2)
        _log_tool("get_extension_quote", inputs, result,
                  duration_ms=int((time.monotonic() - t0) * 1000))
        return result
    except AvisAPIError as e:
        msg = f"Error getting quote: {e.message}"
        _log_tool("get_extension_quote", inputs, msg, success=False,
                  duration_ms=int((time.monotonic() - t0) * 1000))
        return msg
    except AvisAPIUnavailable as e:
        _log_tool("get_extension_quote", inputs, str(e), success=False,
                  duration_ms=int((time.monotonic() - t0) * 1000))
        return str(e)


@function_tool
def extend_rental(reservation_id: str, new_return_datetime: str,
                  email: str, cvv: str, billing_zip: str) -> str:
    """Execute a rental extension. Requires customer verification (email) and
    payment details (CVV and billing zip). Always get a quote first and confirm
    with the customer before calling this."""
    t0 = time.monotonic()
    idem_key = f"{reservation_id}-extend-{uuid.uuid4()}"
    inputs = {"reservation_id": reservation_id, "new_return_datetime": new_return_datetime,
              "email": "***", "has_cvv": True, "billing_zip": billing_zip}
    try:
        data = avis_client.extend_reservation(
            reservation_id, new_return_datetime, email, cvv, billing_zip, idem_key
        )
        result = json.dumps(data, indent=2)
        _log_tool("extend_rental", inputs, result,
                  duration_ms=int((time.monotonic() - t0) * 1000))
        return result
    except AvisAPIError as e:
        msg = f"Extension failed: {e.message}"
        _log_tool("extend_rental", inputs, msg, success=False,
                  duration_ms=int((time.monotonic() - t0) * 1000))
        return msg
    except AvisAPIUnavailable as e:
        _log_tool("extend_rental", inputs, str(e), success=False,
                  duration_ms=int((time.monotonic() - t0) * 1000))
        return str(e)


@function_tool
def cancel_rental(reservation_id: str, email: str, reason: str = "") -> str:
    """Cancel an Avis reservation. Requires customer email for verification.
    Returns cancellation details including any refund or penalty amounts."""
    t0 = time.monotonic()
    idem_key = f"{reservation_id}-cancel-{uuid.uuid4()}"
    inputs = {"reservation_id": reservation_id, "email": "***", "reason": reason}
    try:
        data = avis_client.cancel_reservation(reservation_id, email, reason, idem_key)
        result = json.dumps(data, indent=2)
        _log_tool("cancel_rental", inputs, result,
                  duration_ms=int((time.monotonic() - t0) * 1000))
        return result
    except AvisAPIError as e:
        msg = f"Cancellation failed: {e.message}"
        _log_tool("cancel_rental", inputs, msg, success=False,
                  duration_ms=int((time.monotonic() - t0) * 1000))
        return msg
    except AvisAPIUnavailable as e:
        _log_tool("cancel_rental", inputs, str(e), success=False,
                  duration_ms=int((time.monotonic() - t0) * 1000))
        return str(e)


@function_tool
def check_vehicle_availability(location: str, vehicle_type: str,
                               start_date: str, end_date: str) -> str:
    """Check vehicle availability at a location for a date range.
    Location is a code (e.g. 'LAX'). Dates are YYYY-MM-DD.
    Vehicle types: compact, midsize_sedan, fullsize_sedan, suv, minivan, luxury."""
    try:
        data = avis_client.check_availability(location, vehicle_type, start_date, end_date)
        return json.dumps(data, indent=2)
    except AvisAPIError as e:
        return f"Availability check failed: {e.message}"
    except AvisAPIUnavailable as e:
        return str(e)


# ---------------------------------------------------------------------------
# Agent definition
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are a friendly, professional Avis car rental support agent. You help customers \
with extending and cancelling their rentals.

## What you can do
- **Look up reservations** by ID
- **Extend rentals** — push out the return date/time
- **Cancel reservations** — with refund/penalty details per policy
- **Answer policy questions** using the knowledge base

## How to handle requests

### Extensions
1. Always look up the reservation first to understand the current details.
2. Search the knowledge base for extension policies when relevant.
3. Get a quote for the extension and present the charges to the customer.
4. Only after the customer confirms, collect their email (for verification), CVV, and billing zip.
5. Execute the extension and provide the confirmation number.

### Cancellations
1. Look up the reservation first.
2. Search the knowledge base for cancellation policies and explain any penalties.
3. After the customer confirms they want to proceed, collect their email for verification.
4. Execute the cancellation and provide refund details.

## Important rules
- NEVER fabricate policies — always use search_knowledge_base to look up the answer.
- NEVER execute a write operation without customer confirmation first.
- If the system is temporarily unavailable, apologize and suggest trying again shortly.
- For requests outside your scope (modifications, upgrades, etc.), let the customer know \
those features are coming soon and suggest they contact Avis directly at 1-800-633-3469.
- Be warm, concise, and helpful. Use the customer's name when you know it.
- Always present monetary amounts clearly with currency.
"""

agent = Agent(
    name="Avis Assistant",
    instructions=SYSTEM_PROMPT,
    tools=[
        lookup_reservation,
        search_knowledge_base,
        get_extension_quote,
        extend_rental,
        cancel_rental,
        check_vehicle_availability,
    ],
)
