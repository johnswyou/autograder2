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
from redpen.eval.eval import GoldError, evaluate, markdown, table
from redpen.grade.ingest import IngestSummary, ingest
from redpen.grade.read import problem_of
from redpen.grade.second import vendor_problem
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
    privacy_problems,
    uncached,
)
from redpen.prepare.assemble import assemble
from redpen.prepare.key import (
    KeyFileError,
    constraint_text,
    examine,
    load_key,
    number,
    page_points,
    point_problems,
    save_key,
    spec_point_problems,
    write_markdown,
)
from redpen.prepare.parse import SpecError, missing_numbers, parse
from redpen.prepare.parse import check as check_spec
from redpen.prepare.rubric import write_rubric
from redpen.prepare.selftest import NotEarned, self_test
from redpen.prepare.solve import PartSolution, solve
from redpen.prepare.verify import (
    CHECKER,
    GRADER,
    REFEREE,
    State,
    Verdict,
    Verification,
    changed,
    role_problems,
    state_counts,
    summary,
    verification,
    verification_md,
    verify,
)
from redpen.report.report import review_line, write_report
from redpen.schemas.key import Key, PointConstraint, Policies
from redpen.schemas.spec import Spec
from redpen.schemas.submission import IngestRecord
from redpen.workspace import Workspace, log_failure, read_model, write_model

# An unhandled exception prints no local variable: locals hold the API key and
# student work.
app = typer.Typer(pretty_exceptions_show_locals=False)
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


def _total(text: str) -> PointConstraint:
    return PointConstraint(total=_point_value(text))


def _page_points(text: str) -> PointConstraint:
    return PointConstraint(pages=[_point_value(value) for value in text.split(",")])


# Options on `run` and `prepare`. Given neither, the key's points stay as they are.
Total = Annotated[
    PointConstraint | None,
    typer.Option(
        parser=_total, metavar="POINTS", help="Point total, shared among the parts."
    ),
]
PagePoints = Annotated[
    PointConstraint | None,
    typer.Option(
        parser=_page_points,
        metavar="P1,P2,...",
        help="Points of each page of the blank, in page order.",
    ),
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
    _constraint(total, page_points)  # a dry run leaves the key as it is
    ws = Workspace.for_blank(blank, workspace)
    if dry_run:
        _estimate(ws, blank, _dry_ingest(submissions, roster))
        return
    prepare(blank, total, page_points, profile, roster, False, workspace)
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
    """Build the key only. On a workspace that has one, --total or --page-points
    sets its points and changes nothing else, with no model call."""
    points = _constraint(total, page_points)
    # Refused rather than ignored: ignoring --dry-run would spend money.
    if dry_run:
        _fail("--dry-run is not supported yet")
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
        # Before anything is spent: a lock that could never grade is refused now.
        if found := vendor_problem(lock.models):
            _fail(f"{ws.config_lock_toml}: {found}.")
        if not ws.key_yaml.exists():  # a key will be solved and verified
            _can_verify(ws, lock.models)
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
        if points is not None:
            loaded = loaded.model_copy(update={"points": points})
        problems += _point_problems(ws, loaded)
        for problem in problems:
            typer.echo(problem)
        if problems:
            _fail(f"{ws.key_yaml} has problems and is left as it is.")
        if points is None:
            typer.echo(f"{ws.key_yaml} has no problem and is left as it is.")
            return
        try:
            save_key(ws.key_yaml, loaded)
            write_markdown(ws, loaded)
        except (OSError, KeyFileError) as error:
            _fail(str(error))
        typer.echo(
            f"Set the points of {ws.key_yaml} to {constraint_text(points)}, and"
            f" wrote {ws.solutions_md.name} and {ws.rubric_md.name} again."
        )
        _points(loaded)
        return
    # Before the solver is paid for; the key is checked again once it has criteria.
    if points is not None and (found := spec_point_problems(points, spec, pages)):
        _fail("\n".join(found))
    try:
        key, record = asyncio.run(_key(ws, config, lock, api_key, spec))
        key = key.model_copy(update={"points": points})
        if found := _point_problems(ws, key):
            _fail("\n".join(found))
        save_key(ws.key_yaml, key)
        write_markdown(ws, key)
        _write_verification(ws, record, key)
    except (
        OSError,
        StageError,
        FatalError,
        ResponseValidationError,
        RetryableError,
        KeyFileError,
    ) as error:
        _fail(str(error))
    _states(ws, key, record)
    typer.echo(
        f"Wrote {ws.key_yaml}, {ws.solutions_md.name}, {ws.rubric_md.name},"
        f" {ws.verification_md.name} and {ws.verification_json.name}"
    )
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
) -> tuple[Key, Verification]:
    """Solve, verify, write the rubric from the solution that stands for each part,
    assemble the key, and try its rubric, writing a failed problem's rubric once
    more; with what verification found, for its files."""
    strict = config.strict_privacy
    model, budget = lock.models["solver"], config.image_budget
    async with Gateway(ws.cache_sqlite, api_key, strict_privacy=strict) as gateway:
        solutions = await solve(ws, gateway, model, budget, spec)
        found = await verify(
            ws, gateway, lock.models, budget, spec, solutions, Policies(), False
        )
        standing = {id: v.stands for id, v in found.items()}
        states: dict[str, State] = {id: v.state for id, v in found.items()}
        rubrics = await write_rubric(ws, gateway, model, spec, standing)

        async def again(not_earned: NotEarned) -> Key:
            rubrics.update(
                await write_rubric(ws, gateway, model, spec, standing, not_earned)
            )
            return assemble(spec, standing, rubrics, states)

        first = assemble(spec, standing, rubrics, states)
        key, found = await self_test(
            ws, gateway, lock.models[GRADER], first, found, again
        )
    return key, verification(found, key, lock.models[CHECKER].slug, False)


def _can_verify(ws: Workspace, models: Mapping[str, ModelInfo]) -> None:
    """Exit with one line, before any call, unless `models` has a solver, a
    referee, a grader for the rubric's self-test, and a checker of another vendor
    than the solver's."""
    if found := role_problems(models):
        _fail(f"{ws.config_lock_toml}: {'; '.join(found)}.")


def _write_verification(ws: Workspace, record: Verification, key: Key) -> None:
    write_model(ws.verification_json, record)
    ws.verification_md.write_text(verification_md(record, key), "utf-8")


def _states(ws: Workspace, key: Key, record: Verification) -> int:
    """Say the counts by state and of the self-test, how many parts need a look and
    where it says why, and on stderr how many a failed call left unverified. Returns
    that number. Nothing a model wrote is printed."""
    states = [p.state for p in key.parts]
    typer.echo(f"States: {state_counts(states)}")
    for line in summary(record.parts):
        typer.echo(line)
    look = states.count("unverified")
    typer.echo(f"Parts that need a look: {look}. {ws.verification_md} says why.")
    failed = sum(
        v.ruling == "failed" or bool(v.tried and v.tried[-1].outcome == "not graded")
        for v in record.parts
    )
    if failed:
        typer.echo(
            f"{failed} parts are unverified because a call failed;"
            " `redpen key verify` tries them again.",
            err=True,
        )
    return failed


def _constraint(
    total: PointConstraint | None, pages: PointConstraint | None
) -> PointConstraint | None:
    if total is not None and pages is not None:
        raise typer.BadParameter(
            "cannot be given together with --total", param_hint="'--page-points'"
        )
    return total if total is not None else pages


def _point_problems(ws: Workspace, key: Key) -> list[str]:
    """What stops the key's points being met on the blank of `ws`, one line each."""
    if key.points is None:
        return []
    try:
        with render.open_pdf(ws.blank_pdf) as pdf:
            pages = len(pdf)
    except Exception:  # pdfium reports a file it cannot open as PdfiumError
        return [f"{ws.blank_pdf} cannot be opened to count its pages"]
    return point_problems(key, pages)


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
    if key is not None:
        problems += _point_problems(ws, key)
        for line in _changed_by_hand(ws, key):  # a note, not a problem
            typer.echo(line)
    for problem in problems:
        typer.echo(problem)
    if problems or key is None:
        raise typer.Exit(code=1)
    typer.echo(f"{ws.key_yaml}: no problem found.")
    _points(key)


def _changed_by_hand(ws: Workspace, key: Key) -> list[str]:
    """`changed` against verification.json: nothing without the file, and one line
    when it does not load."""
    if not ws.verification_json.exists():
        return []
    try:
        record = read_model(ws.verification_json, Verification)
    except (OSError, ValueError):  # its text may quote the file
        return [
            f"{ws.verification_json} does not load, so no part is checked for a"
            " change since it was verified."
        ]
    return changed(key, record)


@key_app.command("render")
def key_render(workspace: KeyWorkspace) -> None:
    """Rebuild solutions.md and rubric.md."""
    ws = Workspace(workspace)
    try:
        key = load_key(ws.key_yaml)
    except (OSError, YAMLError, ValidationError):
        _fail(f"{ws.key_yaml} does not load. `redpen key check` lists its problems.")
    if found := _point_problems(ws, key):
        _fail("\n".join(found))
    try:
        write_markdown(ws, key)
    except ArithmeticError:  # a point value past what Decimal can sum
        _fail(
            f"{ws.key_yaml} holds a number too large to work with."
            " `redpen key check` lists its problems."
        )
    typer.echo(f"Wrote {ws.solutions_md} and {ws.rubric_md.name}")


@key_app.command("verify")
def key_verify(workspace: KeyWorkspace) -> None:
    """Solve each problem again with another vendor's model and set each part's
    state. Nothing in key.yaml changes but `state`, and nobody is re-graded."""
    ws = Workspace(workspace)
    try:
        config = load_config()
        api_key = openrouter_api_key()
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
    problems += _point_problems(ws, key)
    for problem in problems:
        typer.echo(problem, err=True)
    if problems:
        _fail(f"{ws.key_yaml} has problems, so it was not verified.")
    _can_verify(ws, lock.models)
    lock = _with_prompts(ws, lock)
    # The key's own answer and solution stand in for the first solver's.
    first = {
        p.id: PartSolution(
            part_id=p.id,
            solution=p.solution,
            answer=p.answer,
            computation="",
            uses=[],
            alternatives=[],
        )
        for p in key.parts
    }

    async def run() -> tuple[Key, dict[str, Verdict]]:
        strict = config.strict_privacy
        async with Gateway(ws.cache_sqlite, api_key, strict_privacy=strict) as gw:
            budget = config.image_budget
            found = await verify(
                ws, gw, lock.models, budget, spec, first, key.policies, True
            )
            # No rewrite: only `state` changes, so no part hash moves.
            return await self_test(ws, gw, lock.models[GRADER], key, found)

    verified, found = asyncio.run(run())
    record = verification(found, verified, lock.models[CHECKER].slug, True)
    try:
        save_key(ws.key_yaml, verified)
        write_markdown(ws, verified)
        _write_verification(ws, record, verified)
    except (OSError, KeyFileError) as error:
        _fail(str(error))
    failed = _states(ws, verified, record)
    typer.echo(
        f"Wrote {ws.key_yaml}, {ws.solutions_md.name}, {ws.verification_md.name} and"
        f" {ws.verification_json.name}"
    )
    if failed:
        raise typer.Exit(code=1)


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
    """Grade the class in `submissions` against the key in `ws`: the check that the
    second reader is another vendor's model, ingest, the check that the pinned
    endpoints keep nothing, then map, read, read again where the reader was unsure,
    and grade, and the report. Exit code 0 only when nothing failed; with
    `dry_run`, an estimate."""
    config, api_key, spec, lock, key = _graded(ws, dry_run)
    if dry_run:
        _estimate(ws, ws.blank_pdf, _dry_ingest(submissions, roster))
        return
    if found := vendor_problem(lock.models):  # before anything is sent
        _fail(f"{ws.config_lock_toml}: {found}.")
    summary = _ingest(ws, submissions, roster)
    lock = _gate(ws, config, lock, api_key)
    failed = _run(ws, spec, key, lock, config, api_key, summary.students, [*STAGES])
    _report(ws, key, summary)
    if summary.failed:
        typer.echo(
            f"Files not ingested: {len(summary.failed)}. {ws.run_report_md} lists"
            " them.",
            err=True,
        )
    if failed or summary.failed:
        raise typer.Exit(code=1)


def _graded(ws: Workspace, dry_run: bool) -> tuple[Config, str, Spec, Lock, Key]:
    """What grading in `ws` needs, or exit with one line before anything is written
    or sent: the configuration, the API key ("" for a dry run), the spec, the lock
    and a key with no problem."""
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
    problems += _point_problems(ws, key)
    for problem in problems:
        typer.echo(problem, err=True)
    if problems:
        _fail(f"{ws.key_yaml} has problems, so nothing was graded.")
    return config, api_key, spec, lock, key


def _gate(ws: Workspace, config: Config, lock: Lock, api_key: str) -> Lock:
    """Exit, with nothing sent, unless the pinned endpoints keep nothing; then the
    lock, written with the prompt versions and the strict privacy of this run."""
    strict = config.strict_privacy
    gate = privacy_problems(lock, api_key, strict)
    if found := asyncio.run(gate):
        for problem in found:
            typer.echo(problem, err=True)
        _fail(
            "Nothing was sent: the pinned endpoints are not verified to keep nothing."
        )
    # Recorded only now, as pages are about to go. Once off it stays off: stored
    # answers may still come from that run, and only a fresh workspace clears it.
    kept = strict and lock.strict_privacy is not False
    lock = _with_prompts(ws, lock.model_copy(update={"strict_privacy": kept}))
    if not strict:
        typer.echo(
            "strict_privacy is off: the endpoints are not checked, and student pages"
            " may go to providers that keep them.",
            err=True,
        )
    return lock


def _run(
    ws: Workspace,
    spec: Spec,
    key: Key,
    lock: Lock,
    config: Config,
    api_key: str,
    students: Sequence[IngestRecord],
    stages: Sequence[str],
) -> list[tuple[str, str]]:
    """`grade_class` over `students` with `stages`, each stage's failures counted on
    stderr; each (stage, student key) that failed."""
    run = grade_class(ws, spec, key, lock.models, config, api_key, students, stages)
    try:
        failed = asyncio.run(run)
    except Exception as error:  # outside any one student: the cache or the ledger
        log_failure(ws, "grade", error)
        _fail(f"The class could not be graded. The traceback is in {ws.run_log}.")
    # Named before the report is tried, so that a report that fails hides none.
    for stage in stages:
        if count := sum(1 for done, _ in failed if done == stage):
            typer.echo(
                f"{stage}: {count} of {len(students)} students failed.", err=True
            )
    if failed:
        typer.echo(f"Each traceback is in {ws.run_log}.", err=True)
    return failed


def _report(ws: Workspace, key: Key, summary: IngestSummary) -> None:
    """Rebuild `out/` for the ingest's students, and say where and how long the
    review list is."""
    try:
        raised = write_report(
            ws, key, summary.students, [f.file for f in summary.failed]
        )
    except Exception as error:  # written to the log: it may quote stored work
        log_failure(ws, "report", error)
        _fail(f"The report could not be written. The traceback is in {ws.run_log}.")
    typer.echo(f"Wrote {ws.root / 'out'}")
    typer.echo(review_line(ws, raised))


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
ESTIMATED = ("parser", "solver", CHECKER, REFEREE, *STAGES.values())


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
    if pinned and not keyed:  # the estimate prices a solve and its verification
        _can_verify(ws, models)
    budget = config.image_budget
    costs = estimate(models, sizes, spec, keyed, summary.students, budget)
    _print_estimate(costs, models, pinned, sizes, spec, keyed, summary.students, budget)
    if found := vendor_problem(models):  # the run itself would be refused
        source = ws.config_lock_toml if pinned else "redpen.toml"
        typer.echo(f"{source}: {found}.", err=True)
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
        " second read, second look or gap re-read; prompt caching, not counted,"
        " costs less.",
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
def regrade(workspace: KeyWorkspace, dry_run: DryRun = False) -> None:
    """Grade the latest ingest again from its stored readings, then audit and
    report: no ingest, map or read. The call cache answers each grade request that
    did not change, so only what a key edit changed is paid for."""
    ws = Workspace(workspace)
    config, api_key, spec, lock, key = _graded(ws, dry_run)
    try:
        summary = read_model(ws.ingest_summary_json, IngestSummary)
    except (OSError, ValueError) as error:
        _fail(f"{ws.root} holds no ingest to grade again ({type(error).__name__}).")
    students = summary.students
    if dry_run:
        if (grader := lock.models.get("grader")) is None:
            _fail(f"{ws.config_lock_toml}: no model is pinned for grader.")
        count = asyncio.run(uncached(ws, key, grader, students))
        typer.echo(f"Grade requests the call cache would not answer: {count}.")
        return
    lock = _gate(ws, config, lock, api_key)
    problems = dict.fromkeys(problem_of(p.id) for p in key.parts)
    unread = [
        s.student_key
        for s in students
        for q in problems
        if not ws.reading_json(s.student_key, q).exists()
    ]
    failed = _run(ws, spec, key, lock, config, api_key, students, ["grade"])
    if unread:
        typer.echo(
            f"Problems not graded, with no stored reading: {len(unread)}, of"
            f" {len(set(unread))} students. {ws.run_report_md} names them.",
            err=True,
        )
    _report(ws, key, summary)
    if failed:
        raise typer.Exit(code=1)


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
    if found := _point_problems(ws, key):
        _fail("\n".join(found))
    _report(ws, key, summary)


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
    workspace: KeyWorkspace,
    top: Annotated[
        int, typer.Option(min=0, help="How many of the largest disagreements to list.")
    ] = 20,
) -> None:
    """Compare the tool's scores with human scores, from stored files. No model
    call. Writes out/eval.md and out/eval.json."""
    ws = Workspace(workspace)
    try:
        key = load_key(ws.key_yaml)
        summary = read_model(ws.ingest_summary_json, IngestSummary)
    except (OSError, YAMLError, ValidationError) as error:
        _fail(
            f"{ws.root} holds no key or no ingest to compare with"
            f" ({type(error).__name__})."
        )
    if found := _point_problems(ws, key):
        _fail("\n".join(found))
    try:
        result = evaluate(ws, key, summary.students, gold, top)
        write_model(ws.eval_json, result)
        ws.eval_md.write_text(markdown(result), encoding="utf-8")
    except GoldError as error:
        _fail(str(error))
    except Exception as error:  # written to the log: it may quote stored work
        log_failure(ws, "eval", error)
        _fail(f"The comparison could not be made. The traceback is in {ws.run_log}.")
    for line in table(result):
        typer.echo(line)
    typer.echo(f"Wrote {ws.eval_md} and {ws.eval_json.name}")


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
