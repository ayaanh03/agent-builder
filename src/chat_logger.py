"""Conversation logger for agent improvement and analysis.

Logs each chat session as a structured JSON file under logs/. Each session
captures: metadata (session id, start/end time, user actions), the full
message history, tool invocations with inputs/outputs, and outcome signals
(errors, escalations, workflow completions).

In production mode (ENVIRONMENT=production in .env), all PII is stripped:
message content is replaced with a placeholder, tool inputs/outputs are
redacted, and only structural metrics are kept.

Usage:
    logger = ChatLogger()
    logger.log_user_message("I'd like to extend my rental.")
    logger.log_agent_message("Sure! What's your reservation ID?")
    logger.log_tool_call("lookup_reservation", {"reservation_id": "AVS-123"}, {...result...})
    logger.finalize()  # writes the session file
"""
import json
import os
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

LOGS_DIR = Path(__file__).resolve().parent.parent / "logs"

# Defaults to production (PII-safe). Set ENVIRONMENT=development in .env
# to enable full debug logging with message content and tool I/O.
_ENV = os.environ.get("ENVIRONMENT", "production").lower().strip()
IS_PRODUCTION = _ENV != "development"

# Module-level active logger — set by the terminal when a session starts.
# Tools can import this and call it without circular deps.
_active_logger: "ChatLogger | None" = None


def get_logger() -> "ChatLogger | None":
    """Get the active session logger, if one exists."""
    return _active_logger


def set_logger(logger: "ChatLogger") -> None:
    """Set the active session logger."""
    global _active_logger
    _active_logger = logger


# ---------------------------------------------------------------------------
# PII scrubbing for production logs
# ---------------------------------------------------------------------------

# Patterns that look like PII in free text
_EMAIL_RE = re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+")
_PHONE_RE = re.compile(r"\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b")
_CVV_FIELD_RE = re.compile(r"\b\d{3,4}\b")  # only used on known CVV fields
_CARD_RE = re.compile(r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b")

# Fields to always redact in tool inputs/outputs (even in structured data)
_SENSITIVE_KEYS = {"email", "cvv", "billing_zip", "card_on_file", "last_four",
                   "customer_name", "license_plate", "address"}


def _redact_text(text: str) -> str:
    """Replace PII patterns in free text with placeholders."""
    text = _EMAIL_RE.sub("[EMAIL]", text)
    text = _PHONE_RE.sub("[PHONE]", text)
    text = _CARD_RE.sub("[CARD]", text)
    return text


def _redact_dict(d: dict) -> dict:
    """Recursively redact sensitive fields from a dict."""
    out = {}
    for k, v in d.items():
        if k.lower() in _SENSITIVE_KEYS:
            out[k] = "[REDACTED]"
        elif isinstance(v, dict):
            out[k] = _redact_dict(v)
        elif isinstance(v, list):
            out[k] = [_redact_dict(i) if isinstance(i, dict) else i for i in v]
        elif isinstance(v, str):
            out[k] = _redact_text(v)
        else:
            out[k] = v
    return out


def _scrub_for_production(value):
    """Scrub a value (str or dict) for production logging."""
    if isinstance(value, dict):
        return _redact_dict(value)
    if isinstance(value, str):
        return _redact_text(value)
    return value


class ChatLogger:
    """Records a single chat session to a structured JSON log file."""

    def __init__(self, session_id: str | None = None):
        self.session_id = session_id or str(uuid.uuid4())[:12]
        self.started_at = datetime.now(timezone.utc).isoformat()
        self.ended_at: str | None = None
        self.production = IS_PRODUCTION
        self.messages: list[dict] = []
        self.tool_calls: list[dict] = []
        self.errors: list[dict] = []
        self.workflows_attempted: list[str] = []
        self.workflows_completed: list[str] = []
        self._message_count = {"user": 0, "agent": 0}
        self._turn = 0

    def log_user_message(self, text: str, source: str = "typed") -> None:
        """Log a user message. source: 'typed', 'option_click', 'option_number', 'option_arrow'."""
        self._turn += 1
        self._message_count["user"] += 1
        entry = {
            "turn": self._turn,
            "role": "user",
            "source": source,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        if self.production:
            entry["content"] = "[redacted]"
            entry["content_length"] = len(text)
        else:
            entry["content"] = text
        self.messages.append(entry)

    def log_agent_message(self, text: str, duration_ms: int | None = None) -> None:
        """Log an agent response. duration_ms is the time the agent took to respond."""
        self._message_count["agent"] += 1
        entry = {
            "turn": self._turn,
            "role": "agent",
            "duration_ms": duration_ms,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        if self.production:
            entry["content"] = "[redacted]"
            entry["content_length"] = len(text)
        else:
            entry["content"] = text
        self.messages.append(entry)

    def log_tool_call(self, tool_name: str, inputs: dict, output: str | dict,
                      success: bool = True, duration_ms: int | None = None) -> None:
        """Log a tool invocation with its inputs and outputs."""
        if self.production:
            logged_inputs = _scrub_for_production(inputs)
            logged_output = _scrub_for_production(output) if isinstance(output, dict) else "[redacted]"
        else:
            logged_inputs = inputs
            logged_output = _truncate(output, max_len=2000)

        self.tool_calls.append({
            "turn": self._turn,
            "tool": tool_name,
            "inputs": logged_inputs,
            "output": logged_output,
            "success": success,
            "duration_ms": duration_ms,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

        # Auto-detect workflow attempts from tool names
        workflow_map = {
            "extend_rental": "extend",
            "get_extension_quote": "extend",
            "cancel_rental": "cancel",
        }
        wf = workflow_map.get(tool_name)
        if wf and wf not in self.workflows_attempted:
            self.workflows_attempted.append(wf)
        if wf and success and tool_name in ("extend_rental", "cancel_rental"):
            if wf not in self.workflows_completed:
                self.workflows_completed.append(wf)

    def log_error(self, error_type: str, message: str, context: dict | None = None) -> None:
        """Log an error that occurred during the session."""
        entry = {
            "turn": self._turn,
            "type": error_type,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        if self.production:
            entry["message"] = _redact_text(message)
            entry["context"] = _scrub_for_production(context) if context else {}
        else:
            entry["message"] = message
            entry["context"] = context or {}
        self.errors.append(entry)

    def finalize(self) -> str | None:
        """Write the session log to disk. Returns the file path, or None on failure."""
        self.ended_at = datetime.now(timezone.utc).isoformat()

        session = {
            "session_id": self.session_id,
            "environment": "production" if self.production else "development",
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "summary": {
                "total_turns": self._turn,
                "user_messages": self._message_count["user"],
                "agent_messages": self._message_count["agent"],
                "tool_calls": len(self.tool_calls),
                "errors": len(self.errors),
                "workflows_attempted": self.workflows_attempted,
                "workflows_completed": self.workflows_completed,
            },
            "messages": self.messages,
            "tool_calls": self.tool_calls,
            "errors": self.errors,
        }

        try:
            LOGS_DIR.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            filename = f"{timestamp}_{self.session_id}.json"
            filepath = LOGS_DIR / filename

            with open(filepath, "w") as f:
                json.dump(session, f, indent=2, ensure_ascii=False)

            return str(filepath)
        except Exception:
            return None


def _truncate(value: str | dict, max_len: int = 2000) -> str | dict:
    """Truncate long string outputs to keep logs manageable."""
    if isinstance(value, dict):
        s = json.dumps(value)
        if len(s) > max_len:
            return {"_truncated": True, "preview": s[:max_len] + "..."}
        return value
    if isinstance(value, str) and len(value) > max_len:
        return value[:max_len] + "... [truncated]"
    return value
