"""`sciknow infer` — manage the llama-server inference substrate.

Subcommands:
    up     [--role writer|embedder|reranker|vlm|all] [--profile P]
    down   [--role …]
    status
    swap   <role> <model.gguf>
    logs   <role> [--n LINES]
"""

from __future__ import annotations

import sys
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from sciknow.config import settings
from sciknow.infer import server as infer_server

app = typer.Typer(
    name="infer",
    help="Manage the llama-server inference substrate (writer/embedder/reranker/vlm).",
    no_args_is_help=True,
)
console = Console()


# Order matters for `--role all`: bring up the cheap roles first
# (embedder + reranker stay tiny) so the writer is the last big load
# and any VRAM error surfaces with the others already up.
# vlm is excluded from `all` because it hot-swaps with the writer
# (both ~17 GB) — `corpus caption-visuals` manages the swap.
_VALID_ROLES = ["writer", "embedder", "reranker", "vlm", "scorer"]
_ALL_BY_DEFAULT = ["writer", "embedder", "reranker"]


def _resolve_roles(role: str) -> list[str]:
    if role == "all":
        return list(_ALL_BY_DEFAULT)
    if role not in _VALID_ROLES:
        console.print(f"[red]Unknown role:[/red] {role!r}. "
                      f"Use one of {_VALID_ROLES + ['all']}")
        raise typer.Exit(2)
    return [role]


@app.command("up")
def up_cmd(
    role: str = typer.Option(
        "writer", "--role", "-r",
        help=f"Role to start. One of {_VALID_ROLES + ['all']}.",
    ),
    profile: str = typer.Option(
        None, "--profile", "-p",
        help="Profile: default | low-vram | spec-dec. "
             "Defaults to settings.infer_profile.",
    ),
    no_wait: bool = typer.Option(
        False, "--no-wait",
        help="Don't block waiting for /health to return 200.",
    ),
):
    """Start one or all llama-server roles."""
    chosen_profile = profile or settings.infer_profile
    roles = _resolve_roles(role)

    for r in roles:
        try:
            info = infer_server.up(r, profile=chosen_profile, wait=not no_wait)
        except Exception as exc:
            console.print(f"[red]✗ {r}: {exc}[/red]")
            raise typer.Exit(1)
        ok = "[green]✓ healthy[/green]" if info.healthy else "[yellow]starting[/yellow]"
        console.print(
            f"{ok} role=[cyan]{r}[/cyan] port={info.port} pid={info.pid} "
            f"model=[dim]{info.model}[/dim]"
        )


@app.command("down")
def down_cmd(
    role: str = typer.Option(
        "all", "--role", "-r",
        help=f"Role to stop. One of {_VALID_ROLES + ['all']}.",
    ),
):
    """Stop one or all llama-server roles."""
    roles = _resolve_roles(role)
    for r in roles:
        stopped = infer_server.down(r)
        if stopped:
            console.print(f"[yellow]⏹ stopped[/yellow] role=[cyan]{r}[/cyan]")
        else:
            console.print(f"[dim]· not running[/dim] role=[cyan]{r}[/cyan]")


@app.command("status")
def status_cmd():
    """Show health + PID + model for every known role."""
    rows = infer_server.status()
    t = Table(title="sciknow infer · status")
    t.add_column("Role", style="cyan")
    t.add_column("Port", justify="right")
    t.add_column("PID", justify="right")
    t.add_column("Health")
    t.add_column("Model", overflow="fold")
    for info in rows:
        h = "[green]✓ healthy[/green]" if info.healthy else "[red]down[/red]"
        t.add_row(info.role, str(info.port), str(info.pid or "—"), h, info.model)
    console.print(t)
    # Exit non-zero if any expected role is down (so scripts can gate on it).
    if not all(info.healthy for info in rows if info.model):
        raise typer.Exit(2)


@app.command("swap")
def swap_cmd(
    role: str = typer.Argument(..., help=f"Role: {_VALID_ROLES}"),
    model: str = typer.Argument(..., help="Path to GGUF or HF id"),
    profile: str = typer.Option(
        None, "--profile", "-p",
        help="Profile to apply on restart.",
    ),
):
    """Replace the role's running model. Down → reconfigure → up."""
    if role not in _VALID_ROLES:
        console.print(f"[red]Invalid role: {role}[/red]")
        raise typer.Exit(2)
    chosen_profile = profile or settings.infer_profile
    info = infer_server.swap(role, model, profile=chosen_profile)
    h = "[green]✓ healthy[/green]" if info.healthy else "[yellow]starting[/yellow]"
    console.print(f"{h} role=[cyan]{role}[/cyan] now model=[dim]{info.model}[/dim]")


@app.command("logs")
def logs_cmd(
    role: str = typer.Argument(..., help=f"Role: {_VALID_ROLES}"),
    n: int = typer.Option(40, "-n", "--lines", help="Tail this many lines."),
):
    """Tail the role's llama-server log file."""
    if role not in _VALID_ROLES:
        console.print(f"[red]Invalid role: {role}[/red]")
        raise typer.Exit(2)
    sys.stdout.write(infer_server.tail_log(role, n=n))
    sys.stdout.write("\n")
