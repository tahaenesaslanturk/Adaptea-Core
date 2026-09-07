from pathlib import Path

from prompt_toolkit.completion import CompleteEvent
from prompt_toolkit.document import Document
from rich.console import Console

from adaptea.repl import COMMANDS, InteractiveRepl, SlashCommandCompleter


def test_slash_command_completer() -> None:
    completer = SlashCommandCompleter()
    event = CompleteEvent()
    doc = Document("/do", cursor_position=3)
    completions = list(completer.get_completions(doc, event))
    assert any(c.text == "/doctor" for c in completions)

    doc_non_slash = Document("hello", cursor_position=5)
    completions_non_slash = list(completer.get_completions(doc_non_slash, event))
    assert len(completions_non_slash) == 0


def test_interactive_repl_init(tmp_path: Path) -> None:
    console = Console(record=True)
    repl = InteractiveRepl(root=tmp_path, console=console)
    assert repl.root == tmp_path.resolve()
    assert "/help" in COMMANDS
    assert "/run" in COMMANDS
    assert "/doctor" in COMMANDS
