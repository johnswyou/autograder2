"""The `redpen` command line (design section 12). Some commands are stubs for now.

Nothing printed to the terminal names a student or repeats what a student or a
model wrote: counts and the workspace's own paths only. The run log in the
workspace holds the tracebacks."""

import asyncio
import contextlib
import logging
import tempfile
import zipfile
from collections import Counter
from collections.abc import Mapping, Sequence
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Annotated, NoReturn

import typer
from pydantic import ValidationError
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table
from ruamel.yaml.error import YAMLError

from redpen import prompts
from redpen.config import (
    Config,
    ConfigError,
    Lock,
    Profile,
    load_config,
    openrouter_api_key,
    pin,
    read_lock,
    write_lock,
)
from redpen.grade.ingest import IngestSummary, ingest
from redpen.imaging import render
from redpen.llm.client import (
    FatalError,
    Gateway,
    ResponseValidationError,
    RetryableError,
    StageError,
)
from redpen.llm.models import (
    ModelError,
    ModelInfo,
    check,
    fetch_catalog,
    recheck_providers,
)
from redpen.orchestrator import (
    GRADE_CROP,
    PATCH,
    READ_REGION,
    STAGES,
    Cost,
    blank_sizes,
    estimate,
    grade_class,
    log_failure,
    privacy_problems,
)
from redpen.prepare.assemble import assemble
from redpen.prepare.key import (
    KeyFileError,
    examine,
    load_key,
    number,
    page_points,
    save_key,
    write_markdown,
)
from redpen.prepare.parse import SpecError, missing_numbers, parse
from redpen.prepare.parse import check as check_spec
from redpen.prepare.rubric import write_rubric
from redpen.prepare.solve import solve
from redpen.report.report import write_report
from redpen.schemas.key import Key
from redpen.schemas.spec import Spec
from redpen.schemas.submission import IngestRecord
from redpen.workspace import Workspace, read_model, write_model

app = typer.Typer()
key_app = typer.Typer(help="Check or render the key.")
models_app = typer.Typer(help="Inspect the configured models.")
app.add_typer(key_app, name="key")
app.add_typer(models_app, name="models")

Blank = Annotated[
    Path, typer.Argument(metavar="BLANK.pdf", help="The blank assignment.")
]
Submissions = Annotated[Path, typer.Argument(help="Folder or ZIP of student PDFs.")]


def _point_value(text: str) -> Decimal:
    """A number of points, read as a Decimal and never through float."""
    try:
        value = Decimal(text.strip())
    except InvalidOperation:
        value = Decimal("NaN")
    if not value.is_finite() or value < 0:
        raise typer.BadParameter(
            f"{text!r} is not a number of points, such as 50 or 2.5"
        )
    return value


# Options on `run` and `prepare`.
Total = Annotated[
    Decimal | None, typer.Option(parser=_point_value, help="Point total.")
]
PagePoints = Annotated[
    Decimal | None, typer.Option(parser=_point_value, help="Points per page.")
]
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
KeyWorkspace = Annotated[
    Path, typer.Option("--workspace", metavar="DIR", help="The assignment's workspace.")
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
    """Everything, start to finish: prepare, grade and report. Safe to run again:
    what was answered before is not paid for twice."""
    for flag, given in [("--total", total), ("--page-points", page_points)]:
        if given is not None:
            _fail(f"{flag} is not supported yet")
    ws = Workspace.for_blank(blank, workspace)
    if dry_run:
        _estimate(ws, blank, _dry_ingest(submissions, roster))
        return
    prepare(blank, None, None, profile, roster, False, workspace)
    _grade(ws, submissions, roster, dry_run=False)


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
    # Refused rather than ignored: ignoring --dry-run would spend money.
    for flag, given in [
        ("--total", total),
        ("--page-points", page_points),
        ("--dry-run", dry_run or None),
    ]:
        if given is not None:
            _fail(f"{flag} is not supported yet")
    ws = Workspace.for_blank(blank, workspace)
    # Opened before anything is copied, so a bad file leaves the workspace free.
    try:
        data = blank.read_bytes()
        with render.open_pdf(data) as pdf:
            pages = len(pdf)
    except Exception as error:  # pdfium reports a file it cannot open as PdfiumError
        _fail(f"{blank} cannot be opened as a PDF: {error}")
    if not pages:
        _fail(f"{blank} has no pages")
    try:
        config = load_config()
        if profile is not None:
            config = config.model_copy(update={"profile": profile})
        api_key = openrouter_api_key()
        if ws.blank_pdf.exists() and ws.blank_pdf.read_bytes() != data:
            _fail(f"{ws.root} holds the key of another blank. Use another --workspace.")
        ws.blank_pdf.parent.mkdir(parents=True, exist_ok=True)
        ws.blank_pdf.write_bytes(data)
        if ws.config_lock_toml.exists():
            try:
                read_lock(ws.config_lock_toml)
            except ValueError as error:  # not TOML or not a lock: its text stays out
                _fail(f"{ws.config_lock_toml} cannot be read ({type(error).__name__}).")
        lock = _with_prompts(ws, asyncio.run(pin(ws, config, api_key)))
        if profile is not None and profile != lock.profile:
            typer.echo(
                f"The workspace's lock pins the profile {lock.profile}, and the lock"
                f" wins over --profile {profile}.",
                err=True,
            )
        if ws.spec_json.exists():
            typer.echo(f"Reusing {ws.spec_json}")
            spec = _checked(ws.spec_json, pages)
        else:
            spec = asyncio.run(_parse(ws, config, lock, api_key))
    except (
        OSError,
        ConfigError,
        ModelError,
        SpecError,
        FatalError,
        ResponseValidationError,
        RetryableError,
    ) as error:
        _fail(str(error))
    if missing := missing_numbers(spec):
        numbers = ", ".join(str(n) for n in missing)
        typer.echo(f"Warning: no problem numbered {numbers} was found.", err=True)
    types = Counter(part.response_type for part in spec.parts)
    typer.echo(f"Pages: {pages}")
    typer.echo(f"Problems: {len({part.id.split('.')[0] for part in spec.parts})}")
    typer.echo(f"Parts: {len(spec.parts)}")
    for response_type, count in sorted(types.items()):
        typer.echo(f"  {response_type}: {count}")
    # The instructor may have edited it: it is checked, never overwritten.
    if ws.key_yaml.exists():
        loaded, problems = examine(ws.key_yaml, spec)
        if loaded is None:
            _does_not_load(ws.key_yaml)
        for problem in problems:
            typer.echo(problem)
        if problems:
            _fail(f"{ws.key_yaml} has problems and is left as it is.")
        typer.echo(f"{ws.key_yaml} has no problem and is left as it is.")
        return
    try:
        key = asyncio.run(_key(ws, config, lock, api_key, spec))
        save_key(ws.key_yaml, key)
        write_markdown(ws, key)
    except (
        OSError,
        StageError,
        FatalError,
        ResponseValidationError,
        RetryableError,
        KeyFileError,
    ) as error:
        _fail(str(error))
    typer.echo(f"Wrote {ws.key_yaml}, {ws.solutions_md.name} and {ws.rubric_md.name}")
    _points(key)


def _with_prompts(ws: Workspace, lock: Lock) -> Lock:
    """`lock` with the version of every prompt file now, written back."""
    lock = lock.model_copy(update={"prompts": prompts.versions()})
    write_lock(ws.config_lock_toml, lock)
    return lock


def _checked(path: Path, pages: int) -> Spec:
    """The spec at `path`, which may have been edited by hand, if it passes the
    checks a parsed spec must pass."""
    try:
        spec = read_model(path, Spec)
    except ValidationError as error:
        # Pydantic's text quotes the file, which a model wrote.
        count = error.error_count()
        raise SpecError(f"{path} is not a valid spec: {count} errors.") from None
    if failures := check_spec(spec, pages):
        raise SpecError(f"{path} fails its checks: {'; '.join(failures)}")
    return spec


async def _parse(ws: Workspace, config: Config, lock: Lock, api_key: str) -> Spec:
    strict = config.strict_privacy
    async with Gateway(ws.cache_sqlite, api_key, strict_privacy=strict) as gateway:
        return await parse(ws, gateway, lock.models["parser"], config.image_budget)


async def _key(
    ws: Workspace, config: Config, lock: Lock, api_key: str, spec: Spec
) -> Key:
    strict = config.strict_privacy
    model = lock.models["solver"]
    async with Gateway(ws.cache_sqlite, api_key, strict_privacy=strict) as gateway:
        solutions = await solve(ws, gateway, model, config.image_budget, spec)
        rubrics = await write_rubric(ws, gateway, model, spec, solutions)
    return assemble(spec, solutions, rubrics)


def _points(key: Key) -> None:
    pages = page_points(key)
    typer.echo(f"Total points: {number(sum(pages.values(), Decimal(0)))}")
    for page, points in pages.items():
        typer.echo(f"  page {page}: {number(points)}")


def _does_not_load(path: Path) -> NoReturn:
    """A key file that does not load, in one line: its problems quote the file,
    and `redpen key check` lists them for whoever edits it."""
    _fail(f"{path} does not load. `redpen key check` lists its problems.")


def _fail(message: str) -> NoReturn:
    typer.echo(message, err=True)
    raise typer.Exit(code=1)


@key_app.command("check")
def key_check(workspace: KeyWorkspace) -> None:
    """Validate key.yaml after a hand edit: schema, ids, dependencies, points."""
    ws = Workspace(workspace)
    try:
        spec = read_model(ws.spec_json, Spec)
    except (OSError, ValidationError) as error:
        _fail(f"{ws.spec_json} cannot be read: {error}")
    key, problems = examine(ws.key_yaml, spec)
    for problem in problems:
        typer.echo(problem)
    if problems or key is None:
        raise typer.Exit(code=1)
    typer.echo(f"{ws.key_yaml}: no problem found.")
    _points(key)


@key_app.command("render")
def key_render(workspace: KeyWorkspace) -> None:
    """Rebuild solutions.md and rubric.md."""
    ws = Workspace(workspace)
    try:
        key = load_key(ws.key_yaml)
    except (OSError, YAMLError, ValidationError):
        _fail(f"{ws.key_yaml} does not load. `redpen key check` lists its problems.")
    try:
        write_markdown(ws, key)
    except ArithmeticError:  # a point value past what Decimal can sum
        _fail(
            f"{ws.key_yaml} holds a number too large to work with."
            " `redpen key check` lists its problems."
        )
    typer.echo(f"Wrote {ws.solutions_md} and {ws.rubric_md.name}")


@app.command()
def grade(
    submissions: Submissions,
    workspace: KeyWorkspace,
    roster: Roster = None,
    dry_run: DryRun = False,
) -> None:
    """Ingest, map, read, grade and report against the existing key."""
    _grade(Workspace(workspace), submissions, roster, dry_run)


def _grade(
    ws: Workspace, submissions: Path, roster: Path | None, dry_run: bool
) -> None:
    """Grade the class in `submissions` against the key in `ws`: ingest, the check
    that the pinned endpoints keep nothing, then map, read and grade, and the
    report. Exit code 0 only when nothing failed; with `dry_run`, an estimate."""
    try:
        config = load_config()
        api_key = "" if dry_run else openrouter_api_key()
    except ConfigError as error:
        _fail(str(error))
    try:
        spec = read_model(ws.spec_json, Spec)
        lock = read_lock(ws.config_lock_toml)
    except (OSError, ValueError) as error:  # missing, or a TOML or schema error
        _fail(f"{ws.root} holds no prepared key ({type(error).__name__}).")
    key, problems = examine(ws.key_yaml, spec)
    if key is None:
        _does_not_load(ws.key_yaml)
    for problem in problems:
        typer.echo(problem, err=True)
    if problems:
        _fail(f"{ws.key_yaml} has problems, so nothing was graded.")
    if dry_run:
        _estimate(ws, ws.blank_pdf, _dry_ingest(submissions, roster))
        return
    summary = _ingest(ws, submissions, roster)
    strict = config.strict_privacy
    lock = _with_prompts(ws, lock.model_copy(update={"strict_privacy": strict}))
    gate = privacy_problems(lock, api_key, strict)
    if found := asyncio.run(gate):
        for problem in found:
            typer.echo(problem, err=True)
        _fail(
            "Nothing was sent: the pinned endpoints are not verified to keep nothing."
        )
    if not strict:
        typer.echo(
            "strict_privacy is off: the endpoints are not checked, and student pages"
            " may go to providers that keep them.",
            err=True,
        )
    students = [s.student_key for s in summary.students]
    run = grade_class(ws, spec, key, lock.models, config, api_key, students)
    try:
        failed = asyncio.run(run)
    except Exception as error:  # outside any one student: the cache or the ledger
        log_failure(ws, "grade", error)
        _fail(f"The class could not be graded. The traceback is in {ws.run_log}.")
    # Named before the report is tried, so that a report that fails hides none.
    for stage in STAGES:
        if count := sum(1 for done, _ in failed if done == stage):
            typer.echo(
                f"{stage}: {count} of {len(students)} students failed.", err=True
            )
    if failed:
        typer.echo(f"Each traceback is in {ws.run_log}.", err=True)
    try:
        write_report(ws, key, summary.students, [f.file for f in summary.failed])
    except Exception as error:  # written to the log: it may quote stored work
        log_failure(ws, "report", error)
        _fail(f"The report could not be written. The traceback is in {ws.run_log}.")
    if summary.failed:
        typer.echo(
            f"Files not ingested: {len(summary.failed)}. {ws.run_report_md} lists"
            " them.",
            err=True,
        )
    typer.echo(f"Wrote {ws.root / 'out'}")
    if failed or summary.failed:
        raise typer.Exit(code=1)


def _ingest(ws: Workspace, submissions: Path, roster: Path | None) -> IngestSummary:
    """Ingest `submissions` and store the summary, whose students every later step
    takes. No student at all is a failure."""
    try:
        summary = ingest(submissions, ws, roster)
    except (OSError, ValueError, zipfile.BadZipFile) as error:
        _fail(
            f"{submissions} cannot be read as a folder or ZIP of PDFs, or the roster"
            f" cannot be read ({type(error).__name__})."
        )
    write_model(ws.ingest_summary_json, summary)
    if not summary.students:
        failed = len(summary.failed)
        _fail(
            f"No student PDF could be ingested from {submissions}: {failed} could"
            " not be read."
            if failed
            else f"No student PDF was found in {submissions}."
        )
    return summary


def _dry_ingest(submissions: Path, roster: Path | None) -> IngestSummary:
    """`submissions` ingested into a folder deleted afterwards, so that a dry run
    leaves the workspace's stored class and its students' files as they were."""
    with tempfile.TemporaryDirectory() as folder:
        return _ingest(Workspace(Path(folder)), submissions, roster)


# The roles the estimate prices.
ESTIMATED = ("parser", "solver", *STAGES.values())


def _estimate(ws: Workspace, blank: Path, summary: IngestSummary) -> None:
    """Print what preparing (unless the key exists) and grading the ingested class
    will cost, with the assumptions. No model is asked anything."""
    try:
        config = load_config()
        if ws.config_lock_toml.exists():
            models, pinned = read_lock(ws.config_lock_toml).models, True
        else:  # listing the models is free
            catalog = asyncio.run(fetch_catalog(openrouter_api_key()))
            names = config.models.model_dump()
            strict = config.strict_privacy
            models = {
                r: catalog.resolve(names[r], strict_privacy=strict) for r in ESTIMATED
            }
            pinned = False
        spec = read_model(ws.spec_json, Spec) if ws.spec_json.exists() else None
    except (ConfigError, ModelError) as error:
        _fail(f"No estimate: {error}")
    except (OSError, ValueError) as error:  # its text may quote the stored file
        _fail(f"No estimate: a stored file cannot be read ({type(error).__name__}).")
    try:
        sizes = blank_sizes(blank)
    except Exception as error:  # pdfium reports a file it cannot open as PdfiumError
        _fail(f"{blank} cannot be opened as a PDF: {error}")
    if not sizes:
        _fail(f"{blank} has no pages")
    keyed = ws.key_yaml.exists()
    budget = config.image_budget
    costs = estimate(models, sizes, spec, keyed, summary.students, budget)
    _print_estimate(costs, models, pinned, sizes, spec, keyed, summary.students, budget)
    if summary.failed:
        _fail(f"Files not ingested: {len(summary.failed)}.")


def _print_estimate(
    costs: Mapping[str, Cost],
    models: Mapping[str, ModelInfo],
    pinned: bool,
    blank: Sequence[tuple[int, int]],
    spec: Spec | None,
    keyed: bool,
    students: Sequence[IngestRecord],
    budget: int,
) -> None:
    total = Cost(
        sum(c.calls for c in costs.values()),
        sum(c.tokens_in for c in costs.values()),
        sum(c.tokens_out for c in costs.values()),
        sum((c.usd for c in costs.values()), Decimal(0)),
    )
    typer.echo("Estimate, with nothing sent to a model:")
    typer.echo(
        f"  {'stage':<8}{'calls':>7}{'tokens in':>12}{'tokens out':>12}{'USD':>9}"
    )
    for stage, c in [*costs.items(), ("total", total)]:
        typer.echo(
            f"  {stage:<8}{c.calls:>7.0f}{c.tokens_in:>12,.0f}"
            f"{c.tokens_out:>12,.0f}{c.usd:>9.4f}"
        )
    pages = sum(len(s.pages) for s in students)
    whose = "pinned in the lock" if pinned else "configured, not pinned yet"
    prices = "; ".join(
        f"{role} {m.slug} {number(m.prompt_price * 10**6)} in,"
        f" {number(m.completion_price * 10**6)} out"
        for role, m in models.items()
        if role in ESTIMATED
    )
    typer.echo("Assumptions:")
    lines = [
        f"{len(students)} students, {pages} pages in all; each answers every part,"
        " so blank or unreadable work costs less than this.",
        f"Images fit to {budget} px, one token per {PATCH} x {PATCH} px; a mapped"
        f" region is {READ_REGION} tokens to the reader and {GRADE_CROP} to the"
        " grader.",
        "Text and output tokens per call as measured on the fixture run. No retry,"
        " second look or gap re-read; prompt caching, not counted, costs less.",
        f"USD per million tokens of the models {whose}: {prices}.",
    ]
    if keyed:
        lines.append("The key exists, so prepare makes no call.")
    elif spec is not None:
        lines.append("The spec exists, so parse makes no call.")
    else:
        lines.append(
            f"No spec yet: prepare is estimated from the blank's {len(blank)} pages,"
            " with the fixture assignment's problems and parts per page, so the"
            " grade estimate is rougher too."
        )
    for line in lines:
        typer.echo(f"- {line}")


@app.command()
def regrade(
    part: Annotated[str | None, typer.Option(help="Only this part.")] = None,
    student: Annotated[str | None, typer.Option(help="Only this student.")] = None,
) -> None:
    """Re-grade whatever a key edit has invalidated."""
    _not_implemented()


@app.command()
def report(workspace: KeyWorkspace) -> None:
    """Rebuild the files in out/ from stored readings and grades. No model call."""
    ws = Workspace(workspace)
    try:
        key = load_key(ws.key_yaml)
        summary = read_model(ws.ingest_summary_json, IngestSummary)
    except (OSError, YAMLError, ValidationError) as error:
        _fail(
            f"{ws.root} holds no key or no ingest to report on"
            f" ({type(error).__name__})."
        )
    try:
        write_report(ws, key, summary.students, [f.file for f in summary.failed])
    except Exception as error:  # written to the log: it may quote stored work
        log_failure(ws, "report", error)
        _fail(f"The report could not be written. The traceback is in {ws.run_log}.")
    typer.echo(f"Wrote {ws.root / 'out'}")


@app.command()
def review() -> None:
    """Step through open flags and accept or override each."""
    _not_implemented()


@app.command()
def override(
    student: str,
    criterion: str,
    points: Annotated[Decimal, typer.Argument(parser=_point_value)],
) -> None:
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
