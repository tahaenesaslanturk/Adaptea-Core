from __future__ import annotations

from pathlib import Path
from typing import Any

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

from adaptea import __version__

ASCII_LOGO = r"""[bold cyan]   ___    ____  ___    ____  ______  _________ 
  /   |  / __ \/   |  / __ \/_  __/ / ____/   |
 / /| | / / / / /| | / /_/ / / /   / __/ / /| |
/ ___ |/ /_/ / ___ |/ ____/ / /   / /___/ ___ |
/_/  |_/_____/_/  |_/_/     /_/   /_____/_/  |_|[/]"""

COMPACT_WORDMARK = "[bold cyan]⚡ ADAPTEA[/] [bold white]v" + __version__ + "[/]"
TAGLINE = "[dim]Adaptive Multi-Model Agent Runtime for Local LLMs[/]"


def format_path(path: Path | str, root: Path | None = None) -> str:
    """Format path relative to root if possible for clean terminal display."""
    p = Path(path)
    if root is not None:
        try:
            rel = p.relative_to(root)
            return f"./{rel}"
        except ValueError:
            pass
    home = Path.home()
    try:
        rel = p.relative_to(home)
        return f"~/{rel}"
    except ValueError:
        return str(p)


def print_banner(
    console: Console,
    root: Path | None = None,
    *,
    compact: bool = False,
    command_name: str | None = None,
) -> None:
    """Render a clean, professional banner for Adaptea CLI."""
    if compact:
        text = Text.from_markup(f"{COMPACT_WORDMARK} [dim]•[/] {TAGLINE}")
        if command_name:
            text.append_text(Text.from_markup(f" [dim]›[/] [bold yellow]{command_name}[/]"))
        console.print(text)
        console.print(Rule(style="dim #334155"))
        return

    console.print(ASCII_LOGO)
    console.print(f" {COMPACT_WORDMARK} [dim]•[/] {TAGLINE}")
    if root is not None:
        console.print(f" [dim]Workspace:[/] [cyan]{format_path(root)}[/]")
    console.print()


def create_panel(
    renderable: Any,
    title: str | None = None,
    subtitle: str | None = None,
    border_style: str = "#334155",
    padding: tuple[int, int] = (1, 2),
) -> Panel:
    """Create a consistent rounded panel."""
    return Panel(
        renderable,
        title=f"[bold cyan]{title}[/]" if title else None,
        subtitle=f"[dim]{subtitle}[/]" if subtitle else None,
        box=box.ROUNDED,
        border_style=border_style,
        padding=padding,
    )


def create_table(*columns: str | tuple[str, dict[str, Any]], title: str | None = None) -> Table:
    """Create a standardized table with rounded borders and subtle headers."""
    table = Table(
        title=f"[bold]{title}[/]" if title else None,
        box=box.ROUNDED,
        header_style="bold cyan",
        border_style="#334155",
        show_edge=True,
        pad_edge=True,
    )
    for col in columns:
        if isinstance(col, tuple):
            name, kwargs = col
            table.add_column(name, **kwargs)
        else:
            table.add_column(col)
    return table
