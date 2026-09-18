"""Textual TUI for the Avis rental support agent.

Single contiguous scrollable chat window. Options appear inline as clickable
text items that also respond to number keys and arrow+enter selection.
"""
from __future__ import annotations

from textual import work, on
from textual.app import App, ComposeResult
from textual.containers import VerticalScroll
from textual.widgets import Footer, Input, Static
from textual.binding import Binding
from textual.message import Message
from textual.reactive import reactive

import time

from agents import Runner

from chat_logger import ChatLogger, set_logger


# ---------------------------------------------------------------------------
# Inline option widget — clickable, highlightable, lives in the chat flow
# ---------------------------------------------------------------------------

class OptionItem(Static):
    """A single selectable option rendered inline in the chat."""

    highlighted = reactive(False)

    class Selected(Message):
        """Fired when this option is chosen (click, enter, or number key)."""
        def __init__(self, text: str) -> None:
            self.text = text
            super().__init__()

    def __init__(self, index: int, label: str, prompt: str) -> None:
        self.index = index
        self.label = label
        self.prompt = prompt
        super().__init__()

    def compose(self) -> ComposeResult:
        return []

    def on_mount(self) -> None:
        self._refresh_display()

    def watch_highlighted(self, value: bool) -> None:
        self._refresh_display()

    def _refresh_display(self) -> None:
        if self.highlighted:
            self.update(f"  [bold reverse] {self.index}. {self.label} [/]")
        else:
            self.update(f"  [bold cyan]{self.index}[/]. {self.label}")

    def on_click(self) -> None:
        self.post_message(self.Selected(self.prompt))


class OptionGroup(Static, can_focus=True):
    """A group of inline options in the chat. Handles arrow keys and number selection."""

    active_index = reactive(0)

    BINDINGS = [
        Binding("up", "move_up", "Previous option", show=False),
        Binding("down", "move_down", "Next option", show=False),
        Binding("enter", "select", "Select option", show=False),
        Binding("1", "pick_1", show=False),
        Binding("2", "pick_2", show=False),
        Binding("3", "pick_3", show=False),
        Binding("4", "pick_4", show=False),
    ]

    def __init__(self, options: list[tuple[str, str]]) -> None:
        """options: list of (label, prompt) tuples."""
        self._options = options
        super().__init__()

    def compose(self) -> ComposeResult:
        for i, (label, prompt) in enumerate(self._options):
            yield OptionItem(i + 1, label, prompt)

    def on_mount(self) -> None:
        self._highlight(0)

    def _highlight(self, idx: int) -> None:
        items = list(self.query(OptionItem))
        if not items:
            return
        idx = max(0, min(idx, len(items) - 1))
        for item in items:
            item.highlighted = False
        items[idx].highlighted = True
        self.active_index = idx

    def action_move_up(self) -> None:
        self._highlight(self.active_index - 1)

    def action_move_down(self) -> None:
        self._highlight(self.active_index + 1)

    def action_select(self) -> None:
        items = list(self.query(OptionItem))
        if items:
            items[self.active_index].post_message(
                OptionItem.Selected(items[self.active_index].prompt)
            )

    def action_pick_1(self) -> None:
        self._select_by_number(1)

    def action_pick_2(self) -> None:
        self._select_by_number(2)

    def action_pick_3(self) -> None:
        self._select_by_number(3)

    def action_pick_4(self) -> None:
        self._select_by_number(4)

    def _select_by_number(self, num: int) -> None:
        items = list(self.query(OptionItem))
        if 0 < num <= len(items):
            items[num - 1].post_message(OptionItem.Selected(items[num - 1].prompt))


# ---------------------------------------------------------------------------
# Chat bubble
# ---------------------------------------------------------------------------

class ChatMessage(Static):
    """A single chat message in the scrollable log."""

    def __init__(self, text: str, is_agent: bool = False) -> None:
        if is_agent:
            markup = f"[bold dodger_blue]🚗 Avis Agent[/]\n{text}"
        else:
            markup = f"[bold green]You[/]\n{text}"
        super().__init__(markup)


# ---------------------------------------------------------------------------
# Menu options
# ---------------------------------------------------------------------------

MENU_OPTIONS = [
    ("🔄 Extend my rental", "I'd like to extend my rental."),
    ("❌ Cancel my reservation", "I need to cancel my reservation."),
    ("🔍 Look up a reservation", "I'd like to look up my reservation."),
    ("❓ Ask about policies", "I have a question about Avis rental policies."),
]


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

class AvisApp(App):
    """Single-window chat TUI for Avis rental support."""

    CSS = """
    Screen {
        layout: vertical;
        background: $surface;
    }

    #chat-scroll {
        height: 1fr;
        padding: 1 2;
    }

    .chat-msg {
        margin: 1 0;
        padding: 0 1;
        height: auto;
    }

    .chat-msg-agent {
        margin: 1 0;
        padding: 0 1;
        height: auto;
    }

    .option-group {
        margin: 0 0 1 2;
        padding: 0;
        height: auto;
    }

    .option-group:focus {
        border: none;
    }

    OptionItem {
        height: auto;
        padding: 0;
        margin: 0;
    }

    OptionItem:hover {
        background: $accent 20%;
    }

    .welcome {
        margin: 1 0 0 0;
        padding: 1 2;
        height: auto;
    }

    .thinking {
        margin: 0 0;
        padding: 0 1;
        height: auto;
        color: $text-muted;
    }

    #user-input {
        dock: bottom;
        margin: 0 2 1 2;
    }
    """

    BINDINGS = [
        Binding("ctrl+c", "quit", "Quit", show=True),
    ]

    TITLE = "Avis Rental Support"

    def __init__(self) -> None:
        super().__init__()
        self.conversation_history: list[dict] = []
        self._options_active = False
        self._msg_source = "typed"  # tracks how the current message was sent
        self._agent_start: float = 0
        self.logger = ChatLogger()
        set_logger(self.logger)

    def compose(self) -> ComposeResult:
        yield VerticalScroll(id="chat-scroll")
        yield Input(placeholder="Type your message or select an option...", id="user-input")
        yield Footer()

    def on_mount(self) -> None:
        """Show welcome message and initial options in the chat."""
        chat = self.query_one("#chat-scroll")

        welcome = Static(
            "[bold dodger_blue]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/]\n"
            "\n"
            "[bold white]  🚗  Welcome to Avis Rental Support[/]\n"
            "\n"
            "  [dim]Hi there! I'm your Avis rental assistant.[/]\n"
            "  [dim]I can help you extend or cancel your rental,[/]\n"
            "  [dim]look up reservations, and answer policy questions.[/]\n"
            "\n"
            "[bold dodger_blue]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/]"
        )
        welcome.add_class("welcome")
        chat.mount(welcome)

        # Show initial menu as an agent message with inline options
        prompt_msg = ChatMessage("How can I help you today?\n", is_agent=True)
        prompt_msg.add_class("chat-msg-agent")
        chat.mount(prompt_msg)

        self._show_options()

    def _show_options(self) -> None:
        """Add inline clickable options to the chat."""
        chat = self.query_one("#chat-scroll")
        group = OptionGroup(MENU_OPTIONS)
        group.add_class("option-group")
        chat.mount(group)
        chat.scroll_end(animate=False)
        self._options_active = True
        group.focus()

    def _dismiss_options(self) -> None:
        """Remove any active option groups."""
        for group in self.query(OptionGroup):
            group.remove()
        self._options_active = False

    @on(OptionItem.Selected)
    def on_option_selected(self, event: OptionItem.Selected) -> None:
        """Handle an option being selected (click, number, or enter)."""
        self._dismiss_options()
        self._msg_source = "option_click"
        self._send_message(event.text)

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle text input submission."""
        text = event.value.strip()
        if not text:
            return
        event.input.value = ""

        # If options are showing and user typed a number, select that option
        if self._options_active and text in ("1", "2", "3", "4"):
            idx = int(text) - 1
            if 0 <= idx < len(MENU_OPTIONS):
                self._dismiss_options()
                self._msg_source = "option_number"
                self._send_message(MENU_OPTIONS[idx][1])
                return

        self._dismiss_options()
        self._msg_source = "typed"
        self._send_message(text)

    def _send_message(self, text: str) -> None:
        """Add user message to chat and trigger agent response."""
        chat = self.query_one("#chat-scroll")

        msg = ChatMessage(text, is_agent=False)
        msg.add_class("chat-msg")
        chat.mount(msg)
        chat.scroll_end(animate=False)

        self.conversation_history.append({"role": "user", "content": text})
        self.logger.log_user_message(text, source=self._msg_source)

        # Show thinking indicator
        thinking = Static("[dim italic]🤔 Thinking...[/]")
        thinking.add_class("thinking")
        thinking.id = "thinking-indicator"
        chat.mount(thinking)
        chat.scroll_end(animate=False)

        # Disable input while processing
        self.query_one("#user-input", Input).disabled = True
        self._agent_start = time.monotonic()

        self._run_agent()

    @work(thread=True)
    def _run_agent(self) -> None:
        """Run the agent in a background thread."""
        from agent import agent

        try:
            result = Runner.run_sync(agent, self.conversation_history)
            response = result.final_output
        except Exception as e:
            response = f"I'm sorry, something went wrong: {e}\nPlease try again."
            self.logger.log_error("agent_exception", str(e))

        duration_ms = int((time.monotonic() - self._agent_start) * 1000)
        self.conversation_history.append({"role": "assistant", "content": response})
        self.logger.log_agent_message(response, duration_ms=duration_ms)
        self.call_from_thread(self._show_response, response)

    def _show_response(self, text: str) -> None:
        """Display agent response."""
        chat = self.query_one("#chat-scroll")

        # Remove thinking indicator
        indicator = self.query("#thinking-indicator")
        for el in indicator:
            el.remove()

        # Add agent response
        msg = ChatMessage(text, is_agent=True)
        msg.add_class("chat-msg-agent")
        chat.mount(msg)
        chat.scroll_end(animate=False)

        # Re-enable input — no options, just free-text chat from here
        inp = self.query_one("#user-input", Input)
        inp.disabled = False
        inp.focus()


    def action_quit(self) -> None:
        """Finalize log and exit immediately."""
        self.logger.finalize()
        self.exit()


def main():
    app = AvisApp()
    app.run()


if __name__ == "__main__":
    main()
