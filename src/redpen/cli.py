"""The `redpen` command line (design section 12). Most commands are stubs for now."""

import asyncio
import contextlib
import logging
from pathlib import Path
from typing import Annotated, NoReturn

import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table

from redpen.config import (
    ConfigError,
    Profile,
    load_config,
    openrouter_api_key,
    read_lock,
)
from redpen.llm.models import ModelError, check, fetch_catalog, recheck_providers
from redpen.workspace import Workspace

app = typer.Typer()
key_app = typer.Typer(help="Check or render the key.")
models_app = typer.Typer(help="Inspect the configured models.")
app.add_typer(key_app, name="key")
app.add_typer(models_app, name="models")

Blank = Annotated[
    Path, typer.Argument(metavar="BLANK.pdf", help="The blank assignment.")
]
Submissions = Annotated[Path, typer.Argument(help="Folder of student PDFs.")]
# Options on `run` and `prepare`.
Total = Annotated[str | None, typer.Option(help="Point total.")]
PagePoints = Annotated[str | None, typer.Option(help="Points per page.")]
ProfileOption = Annotated[
    Profile | None, typer.Option(help="Verification profile. Overrides redpen.toml.")
]
Roster = Annotated[Path | None, typer.Option(metavar="FILE.csv", help="Class roster.")]
DryRun = Annotated[
    bool, typer.Option("--dry-run", help="Print a cost estimate and spend nothing.")
]
WorkspaceDir = Annotated[
    Path | None,
    typer.Option(
        metavar="DIR", help="Workspace. Default: <blank stem>.redpen beside BLANK.pdf."
    ),
]


def _not_implemented() -> NoReturn:
    typer.echo("not implemented yet", err=True)
    raise typer.Exit(code=1)


@app.callback()
def main(
    verbose: Annotated[
        bool, typer.Option("-v", "--verbose", help="Log debug detail.")
    ] = False,
) -> None:
    """Grade student work on an assignment."""
    logging.basicConfig(
        format="%(message)s", handlers=[RichHandler(console=Console(stderr=True))]
    )
    logging.getLogger("redpen").setLevel(logging.DEBUG if verbose else logging.INFO)


@app.command()
def run(
    blank: Blank,
    submissions: Submissions,
    total: Total = None,
    page_points: PagePoints = None,
    profile: ProfileOption = None,
    roster: Roster = None,
    dry_run: DryRun = False,
    workspace: WorkspaceDir = None,
) -> None:
    """Everything, start to finish: prepare, grade, audit, report."""
    _not_implemented()


@app.command()
def prepare(
    blank: Blank,
    total: Total = None,
    page_points: PagePoints = None,
    profile: ProfileOption = None,
    roster: Roster = None,
    dry_run: DryRun = False,
    workspace: WorkspaceDir = None,
) -> None:
    """Build the key only."""
    _not_implemented()


@key_app.command("check")
def key_check() -> None:
    """Validate key.yaml after a hand edit: schema, point sums, page maximums."""
    _not_implemented()


@key_app.command("render")
def key_render() -> None:
    """Rebuild solutions.md and rubric.md."""
    _not_implemented()


@app.command()
def grade(submissions: Submissions) -> None:
    """Ingest, fingerprint, map, read, grade and audit against the existing key."""
    _not_implemented()


@app.command()
def regrade(
    part: Annotated[str | None, typer.Option(help="Only this part.")] = None,
    student: Annotated[str | None, typer.Option(help="Only this student.")] = None,
) -> None:
    """Re-grade whatever a key edit has invalidated."""
    _not_implemented()


@app.command()
def report() -> None:
    """Rebuild the files in out/ from stored grades and overrides."""
    _not_implemented()


@app.command()
def review() -> None:
    """Step through open flags and accept or override each."""
    _not_implemented()


@app.command()
def override(student: str, criterion: str, points: str) -> None:
    """Record a correction."""
    _not_implemented()


@app.command("eval")
def eval_(
    gold: Annotated[
        Path, typer.Option(metavar="FILE.csv", help="Human scores to compare with.")
    ],
) -> None:
    """Compare the tool's scores with human scores."""
    _not_implemented()


@models_app.command("check")
def models_check(
    workspace: Annotated[
        Path | None,
        typer.Option(
            metavar="DIR",
            help="Also re-verify the endpoints pinned in this workspace's lock.",
        ),
    ] = None,
) -> None:
    """Verify the configured models: image input, structured outputs, privacy."""
    try:
        config = load_config()
        api_key = openrouter_api_key()
        catalog = asyncio.run(fetch_catalog(api_key))
    except (ConfigError, ModelError) as error:
        typer.echo(str(error), err=True)
        raise typer.Exit(code=1) from None
    names = config.models.model_dump()
    strict = config.strict_privacy
    problems = check(catalog, names, strict_privacy=strict)
    table = Table("Role", "Configured", "Model", "Endpoints that retain nothing", "")
    for role, name in names.items():
        model, providers = "-", "-"
        with contextlib.suppress(ModelError):
            info = catalog.resolve(name, strict_privacy=strict)
            model = info.slug
            providers = (
                ", ".join(sorted(info.providers)) if strict else "(not required)"
            )
        table.add_row(role, name, model, providers, "FAIL" if problems[role] else "ok")
    Console().print(table)
    found = [problem for each in problems.values() for problem in each]
    lock = Workspace(workspace).config_lock_toml if workspace else None
    if lock is not None and lock.exists():
        pinned = read_lock(lock).models
        rechecked = recheck_providers(pinned, api_key, strict_privacy=strict)
        found += [
            f"{lock}: {p}" for each in asyncio.run(rechecked).values() for p in each
        ]
    for problem in found:
        typer.echo(problem)
    if found:
        raise typer.Exit(code=1)
