"""Avis rental support agent — extend and cancel workflows with RAG.

Exports the `agent` object for use by the terminal UI. No main() here.
Uses SDK built-in guardrails for off-topic detection, built-in tracing for
observability, and SQLiteSession for conversation history (configured in terminal.py).
"""
import os
import uuid
import json
from datetime import datetime, timezone

from dotenv import load_dotenv

load_dotenv()

from agents import (  # noqa: E402
    Agent,
    Runner,
    function_tool,
    InputGuardrail,
    GuardrailFunctionOutput,
    RunContextWrapper,
)

import avis_client  # noqa: E402
from avis_client import AvisAPIError, AvisAPIUnavailable  # noqa: E402
import knowledge_base  # noqa: E402


# ---------------------------------------------------------------------------
# Input guardrail — off-topic detection
# ---------------------------------------------------------------------------

_OFF_TOPIC_KEYWORDS = {
    "code", "program", "script", "python", "javascript", "html",
    "homework", "math", "equation", "solve", "calculate",
    "joke", "funny", "humor", "riddle",
    "recipe", "cook", "weather", "stock", "crypto",
    "write me a", "tell me a joke", "help me with my",
}

_GUARDRAIL_RESPONSE = (
    "I'm only able to help with Avis rental questions — things like extending "
    "or cancelling your reservation, or looking up your rental details. "
    "Is there anything like that I can help you with?"
)

# Guardrail agent — lightweight classifier for off-topic requests
_guardrail_agent = Agent(
    name="Topic Classifier",
    instructions=(
        "You are a classifier that determines if a user message is related to "
        "Avis car rental servicing (extending, cancelling, looking up reservations, "
        "rental policies, vehicle availability). "
        "Respond ONLY with 'on_topic' or 'off_topic'. Nothing else."
    ),
)


async def _check_topic(
    ctx: RunContextWrapper[None],
    agent: Agent,
    input: str | list,
) -> GuardrailFunctionOutput:
    """Run a fast off-topic check on user input.

    Only checks the FIRST message in a conversation. Once there's history,
    follow-up messages (dates, confirmations, "yes", etc.) are almost always
    on-topic and the system prompt handles any edge cases.
    """
    # Skip guardrail when there's conversation history — follow-ups are
    # nearly always on-topic and the system prompt enforces scope anyway
    if isinstance(input, list):
        user_count = sum(1 for item in input
                         if (isinstance(item, dict) and item.get("role") == "user")
                         or (hasattr(item, "role") and item.role == "user"))
        if user_count > 1:
            return GuardrailFunctionOutput(output_info="has_history", tripwire_triggered=False)

        # Extract first user message text
        user_text = ""
        for item in input:
            if isinstance(item, dict) and item.get("role") == "user":
                user_text = item.get("content", "")
                break
            elif hasattr(item, "role") and item.role == "user":
                user_text = getattr(item, "content", "")
                break
        if not user_text:
            return GuardrailFunctionOutput(output_info="no_input", tripwire_triggered=False)
    else:
        user_text = input

    # Quick keyword pre-filter — skip the LLM call for obvious Avis queries
    text_lower = user_text.lower()
    avis_signals = {"reservation", "rental", "avis", "extend", "cancel", "booking",
                    "return", "pickup", "drop off", "dropoff", "avs-", "policy",
                    "vehicle", "car", "suv", "sedan", "minivan", "upgrade",
                    "res ", "look up", "lookup"}
    if any(s in text_lower for s in avis_signals):
        return GuardrailFunctionOutput(output_info="on_topic", tripwire_triggered=False)

    # Use the guardrail agent for ambiguous cases
    result = await Runner.run(_guardrail_agent, user_text)
    is_off_topic = "off_topic" in result.final_output.lower()

    return GuardrailFunctionOutput(
        output_info=result.final_output,
        tripwire_triggered=is_off_topic,
    )


topic_guardrail = InputGuardrail(guardrail_function=_check_topic, name="topic_check")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Fields to strip from write-operation responses. The agent should never see
# expected verification values in tool output — only pass/fail for verification.
_WRITE_REDACT_KEYS = {"email", "cvv", "billing_zip", "customer_email"}

import re

_CVV_RE = re.compile(r"^\d{3,4}$")
_ZIP_RE = re.compile(r"^\d{5}$")


def _validate_payment_fields(cvv: str, billing_zip: str, card_type: str = "") -> str | None:
    """Return an error message if CVV or billing zip format is invalid, else None.

    When card_type is known, enforces the correct CVV length:
    - American Express: exactly 4 digits
    - All others (Visa, Mastercard, Discover, etc.): exactly 3 digits
    """
    if not _CVV_RE.match(cvv):
        return "Invalid CVV — must be exactly 3 or 4 digits."
    if card_type:
        is_amex = "amex" in card_type.lower() or "american express" in card_type.lower()
        if is_amex and len(cvv) != 4:
            return "Invalid CVV — American Express cards require exactly 4 digits."
        if not is_amex and len(cvv) != 3:
            return f"Invalid CVV — {card_type} cards require exactly 3 digits."
    if not _ZIP_RE.match(billing_zip):
        return "Invalid billing zip — must be exactly 5 digits."
    return None


def _sanitize_response(data: dict) -> dict:
    """Strip verification/PII fields from a write API response.

    Keeps charges, confirmation numbers, extension details, etc. — only
    removes fields that could leak expected verification values.
    """
    out = {}
    for k, v in data.items():
        if k in _WRITE_REDACT_KEYS:
            continue
        if isinstance(v, dict):
            out[k] = _sanitize_response(v)
        elif isinstance(v, list):
            out[k] = [_sanitize_response(i) if isinstance(i, dict) else i for i in v]
        else:
            out[k] = v
    return out


def _check_reservation_active(reservation_id: str) -> tuple[str | None, dict | None]:
    """Check if a reservation can be modified.

    Returns (error_message, reservation_data). On success error_message is None
    and reservation_data contains the full reservation (including card_on_file).
    On failure error_message explains why, and reservation_data is None.
    """
    try:
        data = avis_client.get_reservation(reservation_id)
    except AvisAPIError as e:
        return f"Error looking up reservation: {e.message}", None
    except AvisAPIUnavailable as e:
        return str(e), None

    status = data.get("status", "").lower()
    if status != "active":
        return f"This reservation is {status} and can no longer be modified.", None

    return_dt = data.get("dates", {}).get("current_return_datetime", "")
    if return_dt:
        try:
            ret = datetime.fromisoformat(return_dt)
            if ret < datetime.now(timezone.utc):
                local_ret = ret.astimezone()
                return (
                    "This rental's return date has already passed "
                    f"({local_ret.strftime('%A, %B %d at %I:%M %p %Z')}). "
                    "The vehicle has been returned and the "
                    "reservation can no longer be extended or modified."
                ), None
        except (ValueError, TypeError):
            pass  # unparseable date — let the API decide

    return None, data


# ---------------------------------------------------------------------------
# Tools — clean, no manual logging (SDK tracing handles it)
# ---------------------------------------------------------------------------

@function_tool
def lookup_reservation(reservation_id: str) -> str:
    """Look up an Avis reservation by its ID (e.g. 'AVS-29471835').
    Returns reservation details including customer, vehicle, dates, and status."""
    try:
        data = avis_client.get_reservation(reservation_id)
        result = json.dumps(data, indent=2)

        # Flag past-return-date reservations so the agent knows immediately
        return_dt = data.get("dates", {}).get("current_return_datetime", "")
        if return_dt:
            try:
                ret = datetime.fromisoformat(return_dt)
                if ret < datetime.now(timezone.utc):
                    local_ret = ret.astimezone()
                    result += (
                        "\n\n⚠️ NOTE: This rental's return date has already "
                        f"passed ({local_ret.strftime('%A, %B %d at %I:%M %p %Z')}). "
                        "The vehicle has been returned. This reservation "
                        "CANNOT be extended, modified, or cancelled. "
                        "However, membership upgrades are still available."
                    )
            except (ValueError, TypeError):
                pass

        # Surface card info for easy reference when collecting credentials later
        card = data.get("payment", {}).get("card_on_file", {})
        if card.get("type") and card.get("last_four"):
            result += (
                f"\n\n💳 Card on file: {card['type']} ending in {card['last_four']}"
            )

        return result
    except AvisAPIError as e:
        return f"Error looking up reservation: {e.message}"
    except AvisAPIUnavailable as e:
        return str(e)


@function_tool
def search_knowledge_base(query: str) -> str:
    """Search Avis help-center articles for policies, fees, and procedures.
    Use this to answer customer questions about rental policies."""
    results = knowledge_base.search(query, top_k=3)
    if not results:
        return "No relevant articles found for that query."
    parts = []
    for r in results:
        parts.append(f"**{r['title']}** ({r['authority']})\n{r['body']}")
    return "\n\n---\n\n".join(parts)


@function_tool
def get_extension_quote(reservation_id: str, new_return_datetime: str) -> str:
    """Get a price quote for extending a rental to a new return date/time.
    Date format: YYYY-MM-DDTHH:MM:SS with timezone offset (e.g. 2027-06-17T14:00:00-07:00).
    This is a read-only operation — it does not commit the change."""
    err, _ = _check_reservation_active(reservation_id)
    if err:
        return err
    try:
        data = avis_client.get_quote(reservation_id, "extend", new_return_datetime)
        return json.dumps(data, indent=2)
    except AvisAPIError as e:
        return f"Error getting quote: {e.message}"
    except AvisAPIUnavailable as e:
        return str(e)


@function_tool
def extend_rental(reservation_id: str, new_return_datetime: str,
                  email: str, cvv: str, billing_zip: str) -> str:
    """Execute a rental extension. Requires customer verification (email) and
    payment details (CVV and billing zip). Always get a quote first and confirm
    with the customer before calling this."""
    err, res_data = _check_reservation_active(reservation_id)
    if err:
        return err
    card_type = res_data.get("payment", {}).get("card_on_file", {}).get("type", "")
    err = _validate_payment_fields(cvv, billing_zip, card_type)
    if err:
        return err
    idem_key = f"{reservation_id}-extend-{uuid.uuid4()}"
    try:
        data = avis_client.extend_reservation(
            reservation_id, new_return_datetime, email, cvv, billing_zip, idem_key
        )
        return json.dumps(_sanitize_response(data), indent=2)
    except AvisAPIError as e:
        if e.code == "VERIFICATION_FAILED":
            return "Verification failed. The email provided does not match our records."
        if e.code == "PAYMENT_VALIDATION_ERROR":
            return "Payment verification failed. Please double-check the CVV and billing zip."
        if e.code == "PAYMENT_DECLINED":
            return "The payment was declined. Please verify your card details or try a different card."
        return f"Extension failed: {e.message}"
    except AvisAPIUnavailable as e:
        return str(e)


@function_tool
def cancel_rental(reservation_id: str, email: str, reason: str = "") -> str:
    """Cancel an Avis reservation. Requires customer email for verification.
    Returns cancellation details including any refund or penalty amounts."""
    err, _ = _check_reservation_active(reservation_id)
    if err:
        return err
    idem_key = f"{reservation_id}-cancel-{uuid.uuid4()}"
    try:
        data = avis_client.cancel_reservation(reservation_id, email, reason, idem_key)
        return json.dumps(_sanitize_response(data), indent=2)
    except AvisAPIError as e:
        if e.code == "VERIFICATION_FAILED":
            return "Verification failed. The email provided does not match our records."
        return f"Cancellation failed: {e.message}"
    except AvisAPIUnavailable as e:
        return str(e)


@function_tool
def get_modification_quote(reservation_id: str, new_return_datetime: str,
                           new_return_location: str = "") -> str:
    """Get a price quote for modifying a rental (changing return time or location).
    Date format: YYYY-MM-DDTHH:MM:SS with timezone offset.
    This is a read-only operation — it does not commit the change."""
    err, _ = _check_reservation_active(reservation_id)
    if err:
        return err
    try:
        data = avis_client.get_quote(
            reservation_id, "modify", new_return_datetime,
            new_return_location=new_return_location or None,
        )
        return json.dumps(data, indent=2)
    except AvisAPIError as e:
        return f"Error getting quote: {e.message}"
    except AvisAPIUnavailable as e:
        return str(e)


@function_tool
def modify_rental(reservation_id: str, email: str, cvv: str, billing_zip: str,
                  new_pickup_datetime: str = "", new_return_datetime: str = "",
                  new_return_location: str = "") -> str:
    """Modify a reservation — change pickup/return time or return location.
    Requires email verification and payment details (CVV + billing zip).
    At least one of new_pickup_datetime, new_return_datetime, or new_return_location
    must be provided. Always get a quote first and confirm with the customer."""
    err, res_data = _check_reservation_active(reservation_id)
    if err:
        return err
    card_type = res_data.get("payment", {}).get("card_on_file", {}).get("type", "")
    err = _validate_payment_fields(cvv, billing_zip, card_type)
    if err:
        return err
    if not any([new_pickup_datetime, new_return_datetime, new_return_location]):
        return "At least one change must be specified (new pickup time, return time, or return location)."
    idem_key = f"{reservation_id}-modify-{uuid.uuid4()}"
    try:
        data = avis_client.modify_reservation(
            reservation_id, email, cvv, billing_zip,
            new_pickup_datetime=new_pickup_datetime or None,
            new_return_datetime=new_return_datetime or None,
            new_return_location=new_return_location or None,
            idempotency_key=idem_key,
        )
        return json.dumps(_sanitize_response(data), indent=2)
    except AvisAPIError as e:
        if e.code == "VERIFICATION_FAILED":
            return "Verification failed. The email provided does not match our records."
        if e.code == "PAYMENT_VALIDATION_ERROR":
            return "Payment verification failed. Please double-check the CVV and billing zip."
        if e.code == "PAYMENT_DECLINED":
            return "The payment was declined. Please verify your card details or try a different card."
        if e.code == "VEHICLE_UNAVAILABLE":
            return (
                "Unfortunately, that location cannot accept your vehicle type for return. "
                "Use check_vehicle_availability to find nearby locations that can accept "
                "this vehicle type, or direct the customer to Avis Customer Service at "
                "1-800-XXX-XXXX."
            )
        return f"Modification failed: {e.message}"
    except AvisAPIUnavailable as e:
        return str(e)


@function_tool
def upgrade_membership(reservation_id: str, email: str) -> str:
    """Upgrade a standard customer to Avis Preferred membership.
    Requires the customer_id (from reservation lookup) and email verification.
    Only works for customers with 'standard' membership — 'avis_preferred' members
    are already upgraded."""
    # Look up reservation to get customer_id
    try:
        res = avis_client.get_reservation(reservation_id)
    except AvisAPIError as e:
        return f"Error looking up reservation: {e.message}"
    except AvisAPIUnavailable as e:
        return str(e)

    if res.get("membership_status") == "avis_preferred":
        return "This customer is already an Avis Preferred member."

    customer_id = res.get("customer_id")
    if not customer_id:
        return "Could not determine the customer ID from this reservation."

    idem_key = f"{customer_id}-upgrade-{uuid.uuid4()}"
    try:
        data = avis_client.upgrade_customer(customer_id, email, idem_key)
        return json.dumps(_sanitize_response(data), indent=2)
    except AvisAPIError as e:
        if e.code == "VERIFICATION_FAILED":
            return "Verification failed. The email provided does not match our records."
        if e.code == "ALREADY_PREFERRED":
            return "This customer is already an Avis Preferred member."
        if e.code == "NOT_ELIGIBLE":
            return "This customer is not currently eligible for an upgrade to Avis Preferred."
        return f"Upgrade failed: {e.message}"
    except AvisAPIUnavailable as e:
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

def _get_local_tz_name() -> str:
    """Detect the user's local timezone for display purposes."""
    try:
        local_now = datetime.now().astimezone()
        return local_now.strftime("%Z")  # e.g. "PDT", "EST", "CST"
    except Exception:
        return "local time"


_TZ_NAME = _get_local_tz_name()


def _build_system_prompt(ctx: RunContextWrapper, ag: Agent) -> str:
    """Build the system prompt with the current datetime.

    Called dynamically by the SDK before each agent run so the agent
    always has an accurate clock — prevents it from guessing whether
    a return date is in the past.
    """
    now_local = datetime.now().astimezone()
    now_str = now_local.strftime("%A, %B %d, %Y at %I:%M %p") + f" {_TZ_NAME}"

    return f"""\
You are a friendly, professional Avis car rental support agent. You ONLY help customers \
with their Avis rentals — extending, cancelling, looking up reservations, and answering \
Avis rental policy questions. You do NOT help with anything else.

## Scope guardrails
- You MUST decline any request that is not related to Avis car rentals.
- If a customer asks you to write code, do math homework, tell jokes, give travel advice, \
or anything outside of Avis rental servicing, politely redirect: \
"I'm only able to help with Avis rental questions — things like extending or cancelling \
your reservation, or looking up your rental details. Is there anything like that I can \
help you with?"
- NEVER answer off-topic questions, even if you know the answer. Stay in character.

## Current time
The current date and time is **{now_str}**.

## Timezone
The customer's local timezone is **{_TZ_NAME}**. When presenting ANY dates \
or times to the customer — whether from a reservation lookup, a quote, an extension \
confirmation, or any other source — you MUST convert them from whatever timezone they \
arrive in (often UTC+00:00) to **{_TZ_NAME}** and display in a human-friendly \
format like "Sunday, June 15 at 2:00 PM {_TZ_NAME}". \
NEVER show raw ISO timestamps, UTC offsets, or "+00:00" times to the customer.

## Past-date determination
Do NOT guess whether a reservation's return date has passed. The tools will include a \
⚠️ warning if the return date is in the past. If no warning is present, the reservation \
is still active and eligible for changes.

## Membership acknowledgment
When you look up a reservation and the customer has `membership_status: "avis_preferred"`, \
thank them for their loyalty — e.g. "Thank you for being an Avis Preferred member!" Do this \
naturally as part of presenting the reservation details, not as a separate message.

## What you can do
- **Look up reservations** by ID
- **Extend rentals** — push out the return date/time
- **Modify rentals** — change pickup/return time or return location
- **Cancel reservations** — with refund/penalty details per policy
- **Upgrade membership** — upgrade standard customers to Avis Preferred
- **Answer policy questions** using the knowledge base

## How to handle requests

### Extensions
1. Always look up the reservation first to understand the current details.
2. **Check eligibility**: the reservation must have status "active" AND the current return \
date must be in the future. If the car has already been returned or the reservation is \
completed/cancelled, tell the customer the rental has already ended and cannot be extended.
3. Search the knowledge base for extension policies when relevant.
4. If the customer provides a date (e.g. "october 12", "next Friday"), treat it as the \
desired new return date and get a quote for it. Do NOT ask them to repeat the date.
5. Present the quote charges to the customer. The quote includes `extension_days` — compare it \
to the previous quote or to what the customer might expect. **Billing is by full rental days**: \
even one hour past the day boundary costs a full extra day. If the customer's chosen time \
pushes them into an extra day (e.g. returning at 1:00 AM instead of 12:00 AM), proactively \
point this out and mention they could save money by returning at the original time-of-day.
6. Only after the customer confirms, collect their email (for verification), CVV, and billing zip. \
When asking for the CVV, mention the card type and last 4 digits from `card_on_file` so they \
know which card to use (e.g. "Please provide the CVV for your Visa ending in 4832").
7. Execute the extension and provide the confirmation number.

### Modifications (change time or location)
1. Always look up the reservation first to understand the current details.
2. **Check eligibility**: same rules as extensions — status "active" and return date in the future.
3. Clarify what the customer wants to change: pickup time, return time, return location, or a combination.
4. **Validate locations early**: if the customer gives a city name (e.g. "NYC", "LA", "Chicago") instead \
of a specific Avis location code, use `check_vehicle_availability` with the nearest major airport code \
for that city to find available locations. The API response includes `nearby_locations` which lists other \
Avis branches in the area. Present all available options (the searched location plus any nearby ones) and \
let the customer pick. Do NOT guess or suggest location codes you haven't verified — only suggest locations \
that the availability API actually returned.
5. Get a modification quote and present the charges. A return-location change may incur a one-way fee.
6. Only after the customer confirms, collect their email (for verification), CVV, and billing zip. \
When asking for the CVV, mention the card type and last 4 digits from `card_on_file`.
7. Execute the modification and provide the confirmation details.

### Cancellations
1. Look up the reservation first.
2. **Check eligibility**: the reservation must have status "active". If it's already \
completed, cancelled, or the return date has passed, inform the customer accordingly.
3. Search the knowledge base for cancellation policies and explain any penalties.
4. After the customer confirms they want to proceed, collect their email for verification.
5. Execute the cancellation and provide refund details.

### Upgrades (standard → Avis Preferred)
Upgrades are about the **customer**, not the rental — they do NOT require an active reservation. \
Even if the return date has passed or the reservation is completed, the customer can still upgrade.
1. Look up the reservation to check the customer's current `membership_status`.
2. If already `avis_preferred`, let them know they're already a Preferred member.
3. If `standard`, ALWAYS use search_knowledge_base to look up Avis Preferred benefits and \
present them to the customer BEFORE asking if they want to proceed. The customer needs to know \
what they're signing up for (e.g. late-fee waivers, priority vehicle access, counter bypass).
4. After presenting the benefits, ask if they'd like to proceed. Only THEN collect their email \
for verification. Let them know that **only their email is needed** — no CVV or billing zip required.
5. Execute the upgrade using the `customer_id` from the reservation lookup.

## Conversation style
- When the customer provides information in context (a date, a confirmation, a reservation \
ID), connect it to the current workflow. Don't ask them to repeat themselves.
- Short follow-ups like "yes", "sure", a date, or an ID are responses to YOUR last question — \
treat them as such.
- Be warm, concise, and helpful. Use the customer's name when you know it.
- Always present monetary amounts clearly with currency.
- Always use `current_return_datetime` as the customer's return date — never reference \
`original_return_datetime`. If the customer says "same time" or "keep the current date," use \
`current_return_datetime` without asking which date they mean.
- **Validate dates and times**: if the customer provides something ambiguous or impossible \
(e.g. "13pm", "February 30", "next Blursday"), ask them to clarify before proceeding. \
Common typos like "13pm" likely mean "1pm" — suggest the correction and confirm.

## Verification — DO NOT second-guess customer input
When the customer provides their email, CVV, or billing zip, pass the values EXACTLY as given \
to the tool. Do NOT:
- Spell-check or question email domains (e.g. "exmaple.com" could be a legitimate domain — \
you have no way to know)
- Suggest the customer may have made a typo in their email
- Refuse to proceed because an email "looks wrong"
The verification system will accept or reject the email — your job is to relay, not validate.

Customers often provide all credentials in a single message in any order, e.g. \
"john@example.com 847 90210" or "90210 john@example.com 847". Identify each value by its format: \
the email has an @, the CVV is 3-4 digits, and the billing zip is 5 digits. Parse ALL values \
from the message and proceed immediately. Do NOT ask for any value the customer already provided.

## Card info for credential collection
When asking the customer for their CVV, you MUST reference the **actual** card type and last 4 \
digits from the reservation's `payment.card_on_file` field — e.g. "the CVV for your Mastercard \
ending in 2941". NEVER use placeholders like "[card type]" or "[last 4 digits]". If you don't \
remember the card info, look up the reservation again before asking.

## Escalation
When you can't help with something (e.g. membership downgrades, billing disputes, complex \
account issues), direct the customer to Avis Customer Service at **1-800-XXX-XXXX**. Always \
provide this number — never tell the customer to "contact Avis" without giving them a way to do so.

## Important rules
- NEVER fabricate policies — always use search_knowledge_base to look up the answer.
- NEVER execute a write operation without customer confirmation first.
- Trust the data returned by tools. If a quote returns a price, present it as-is — do NOT \
question, second-guess, or refuse to proceed because a price seems surprising. Pricing is \
calculated by the system (e.g. daily rates mean different times on the same day cost the same). \
Your job is to relay the information, not audit it.
- If the system is temporarily unavailable, apologize and suggest trying again shortly.
"""


agent = Agent(
    name="Avis Assistant",
    instructions=_build_system_prompt,
    tools=[
        lookup_reservation,
        search_knowledge_base,
        get_extension_quote,
        extend_rental,
        get_modification_quote,
        modify_rental,
        cancel_rental,
        upgrade_membership,
        check_vehicle_availability,
    ],
    input_guardrails=[topic_guardrail],
)
