# Avis Rental Support Agent

An AI-powered customer support agent for Avis car rental servicing, built on the OpenAI Agents SDK with a Textual TUI.

## Workflow Coverage

I chose to implement **all four workflows** — extend, modify, cancel, and upgrade — because they share enough infrastructure (reservation lookup, eligibility checks, verification, payment validation) that each incremental workflow was relatively cheap to add once the foundation was solid. The harder problem was getting the foundation right: security, validation, error handling, and a good conversational UX.

### What's implemented

| Workflow | Description |
|----------|-------------|
| **Extend** | Push out the return date. Quote → confirm → verify (email + CVV + zip) → execute. |
| **Modify** | Change return time or location. Validates location codes via the availability API before quoting. Handles `VEHICLE_UNAVAILABLE` by searching nearby locations. |
| **Cancel** | Cancel with refund/penalty details from the knowledge base (no cancel-quote endpoint exists). Email-only verification. |
| **Upgrade** | Standard → Avis Preferred. Email-only, no CVV/zip. Works even on expired reservations since upgrades are about the customer, not the rental. |

### What I intentionally left out

- **Reservation lookup by email** — the API only supports lookup by reservation ID. No workaround possible.
- **Vehicle type changes** — the modify API only supports time/location changes, not vehicle swaps.
- **Membership downgrade** — no API endpoint. Agent directs to Avis Customer Service.

## Design Decisions

### Security: Black-Box Verification

The agent **never sees expected verification values**. The customer's email isn't returned by `GET /reservations`, so the agent can't leak it even under prompt injection. Write tool responses are sanitized (`_sanitize_response`) to strip `email`, `cvv`, `billing_zip`, and `customer_email` at all nesting levels. Verification failures return generic messages ("The email provided does not match our records") with no hints.

### Card-Type-Aware CVV Validation

CVV length is validated against the card type from the reservation: Amex requires exactly 4 digits, all others (Visa, Mastercard, Discover) require exactly 3. This catches mismatches before the API call. Card type is obtained from `_check_reservation_active()`, which returns a `(error, reservation_data)` tuple — no extra API call needed.

### Three-Layer Eligibility Checks

1. `lookup_reservation` appends a ⚠️ warning if the return date has passed (agent sees it inline)
2. `_check_reservation_active()` gates all quote and write tools — checks status AND return date against current UTC time
3. The API itself returns `409 RESERVATION_NOT_ACTIVE` as a final safety net

The mock API may report `status: "active"` for returned cars, so the return-date comparison is the reliable signal.

### Dynamic System Prompt

The agent's instructions are a callable (`_build_system_prompt`) invoked before each run. It injects the current datetime so the agent has an accurate clock and never guesses whether a date is in the past. It also enforces: always use `current_return_datetime` (never `original`), thank Preferred members, validate locations via the availability API (never guess codes), parse credentials in any order, never spell-check customer emails, and catch invalid times like "13pm".

### Location Discovery

When a customer gives a city name ("NYC", "LA"), the agent uses `check_vehicle_availability` with the nearest major airport code. The API's `nearby_locations` field discovers actual Avis branches. The agent never suggests unverified location codes — the mock API has a limited location set (JFK, LAX, SFO, ORD, DFW, SNA, SJC, BUR, LGB, OAK).

### RAG

Pure-Python TF-IDF with cosine similarity over 30 knowledge-base articles. No scikit-learn (PyPy compatibility). `official-policy` articles get a 1.15× relevance boost. ~50 lines, zero external dependencies. Used for policy questions, cancellation penalties (no cancel-quote endpoint), and upgrade benefits.

### API Resilience

- **Retries**: tenacity with exponential backoff (0.5→1→2s) + jitter, 3 attempts, only on 5xx and timeouts
- **Idempotency keys**: `{reservation_id}-{action}-{uuid4}`, reused across retries
- **Custom exceptions**: `AvisAPIError` (4xx, never retried) vs `AvisAPIUnavailable` (exhausted retries)
- **Timeouts**: 10s reads, 20s writes

### TUI

Textual framework with a single scrollable chat window. Six menu options (extend, modify, cancel, upgrade, lookup, policies) with number key bindings. Markdown-to-Rich conversion (`_md_to_rich`) handles the LLM's `**bold**` output. 500-character input limit. Generic error messages — internal errors never leak to the user.

### Guardrail

SDK `InputGuardrail` with a lightweight classifier agent, scoped to the first message only. Once there's conversation history, follow-ups pass through (the system prompt handles scope enforcement). A keyword pre-filter skips the LLM classifier for obvious Avis queries.

## How to Run

```bash
# 1. Create and activate a virtual environment (Python 3.10+ / PyPy 3.11)
python3 -m venv .venv
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure environment
cp env.example .env
# Fill in AVIS_API_KEY and OPENAI_API_KEY in .env

# 4. Verify API connection
python src/avis_client.py

# 5. Run the agent
python src/main.py
```

## Logs & Observability

- **OpenAI Traces**: All LLM calls, tool invocations, and guardrail checks are automatically traced via the SDK. View at [platform.openai.com/traces](https://platform.openai.com/traces). Set `ENVIRONMENT=development` in `.env` to include full request/response data in traces (production mode strips PII).
- **`src/chat_logger.py`**: A structured JSON session logger with PII redaction is available but not wired in — the SDK's built-in tracing covers the same ground with less code.

## File Structure

```
src/
├── main.py            # Entry point
├── terminal.py        # Textual TUI (chat UI, options, Markdown→Rich)
├── agent.py           # Agent definition (9 tools, dynamic prompt, guardrail, validation)
├── avis_client.py     # HTTP client (7 endpoints, retries, idempotency)
├── knowledge_base.py  # TF-IDF RAG over help articles
└── chat_logger.py     # Legacy session logger (not wired in)

data/knowledge-base/articles.json   # 30 help-center articles
docs/api-reference.md               # Full API documentation
```
