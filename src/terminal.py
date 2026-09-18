"""Textual TUI for the Avis rental support agent.

Provides a mouse-clickable interface with action buttons, a scrollable chat
log, and a text input field. Run via `python src/main.py`.
"""
from __future__ import annotations

from textual import work
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import Button, Footer, Header, Input, Static, RichLog
from textual.binding import Binding

from agents import Runner


# ---------------------------------------------------------------------------
# Chat message widget
# ---------------------------------------------------------------------------

class ChatMessage(Static):
    """A single chat message bubble."""

    def __init__(self, sender: str, text: str, is_agent: bool = False) -> None:
        if is_agent:
            markup = f"[bold dodger_blue]🚗 Avis Agent[/]\n{text}"
        else:
            markup = f"[bold green]You[/]\n{text}"
        super().__init__(markup)


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------

WELCOME_TEXT = """\
[bold dodger_blue]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/]

[bold white]  🚗  Welcome to Avis Rental Support[/]

[dim]  I'm here to help you manage your rental. I can:[/]
[dim]  • Extend your rental        • Cancel a reservation[/]
[dim]  • Look up reservation info  • Answer policy questions[/]

[dim]  Click a button below or type your question![/]

[bold dodger_blue]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/]
"""

# Button prompts — clicking sends these as user messages
BUTTON_PROMPTS = {
    "btn_extend": "I'd like to extend my rental.",
    "btn_cancel": "I need to cancel my reservation.",
    "btn_lookup": "I'd like to look up my reservation.",
    "btn_policy": "I have a question about Avis rental policies.",
}


class AvisApp(App):
    """The Avis rental support TUI."""

    CSS = """
    Screen {
        layout: vertical;
    }

    #welcome {
        height: auto;
        padding: 1 2;
    }

    #chat-scroll {
        height: 1fr;
        border: solid dodgerblue;
        padding: 0 1;
    }

    .chat-msg {
        margin: 1 0;
        padding: 1 2;
        height: auto;
    }

    .chat-msg.agent {
        background: $surface;
    }

    #button-bar {
        height: auto;
        padding: 1 1;
        align: center middle;
    }

    #button-bar Button {
        margin: 0 1;
        min-width: 24;
    }

    #input-bar {
        height: auto;
        padding: 1 2;
    }

    #user-input {
        width: 1fr;
    }

    #thinking {
        height: auto;
        padding: 0 2;
        color: $text-muted;
    }
    """

    BINDINGS = [
        Binding("ctrl+c", "quit", "Quit", show=True),
    ]

    TITLE = "Avis Rental Support"

    def __init__(self) -> None:
        super().__init__()
        self.conversation_history: list[dict] = []

    def compose(self) -> ComposeResult:
        yield Header()
        yield Static(WELCOME_TEXT, id="welcome")
        yield VerticalScroll(id="chat-scroll")
        with Horizontal(id="button-bar"):
            yield Button("🔄 Extend Rental", id="btn_extend", variant="primary")
            yield Button("❌ Cancel Reservation", id="btn_cancel", variant="error")
            yield Button("🔍 Look Up", id="btn_lookup", variant="default")
            yield Button("❓ Policies", id="btn_policy", variant="default")
        yield Static("", id="thinking")
        yield Input(placeholder="Type your message here...", id="user-input")
        yield Footer()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button clicks — send the corresponding prompt."""
        prompt = BUTTON_PROMPTS.get(event.button.id, "")
        if prompt:
            self._send_message(prompt)

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle Enter key in the input field."""
        text = event.value.strip()
        if text:
            event.input.value = ""
            self._send_message(text)

    def _send_message(self, text: str) -> None:
        """Add user message to chat and trigger agent response."""
        # Add user message to UI
        chat = self.query_one("#chat-scroll")
        msg = ChatMessage("You", text, is_agent=False)
        msg.add_class("chat-msg")
        chat.mount(msg)
        chat.scroll_end(animate=False)

        # Add to conversation history
        self.conversation_history.append({"role": "user", "content": text})

        # Show thinking indicator and disable input
        self.query_one("#thinking", Static).update("[dim italic]🤔 Thinking...[/]")
        self.query_one("#user-input", Input).disabled = True
        for btn in self.query("Button"):
            btn.disabled = True

        # Run agent in background worker
        self._run_agent()

    @work(thread=True)
    def _run_agent(self) -> None:
        """Run the agent in a background thread to keep the UI responsive."""
        from agent import agent  # import here to avoid circular imports

        try:
            result = Runner.run_sync(agent, self.conversation_history)
            response = result.final_output
        except Exception as e:
            response = f"I'm sorry, something went wrong: {e}\nPlease try again."

        self.conversation_history.append({"role": "assistant", "content": response})
        self.call_from_thread(self._show_response, response)

    def _show_response(self, text: str) -> None:
        """Display agent response in the chat (called from main thread)."""
        chat = self.query_one("#chat-scroll")
        msg = ChatMessage("Agent", text, is_agent=True)
        msg.add_class("chat-msg", "agent")
        chat.mount(msg)
        chat.scroll_end(animate=False)

        # Clear thinking indicator and re-enable input
        self.query_one("#thinking", Static).update("")
        self.query_one("#user-input", Input).disabled = False
        for btn in self.query("Button"):
            btn.disabled = False
        self.query_one("#user-input", Input).focus()


def main():
    app = AvisApp()
    app.run()


if __name__ == "__main__":
    main()
