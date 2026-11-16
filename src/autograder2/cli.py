"""The `autograder2` command line.

Nothing printed to the terminal names a student or repeats what a student or a
model wrote: counts and the workspace's own paths only. The one exception is
`autograder2 review`, whose purpose is to show it, on stdout alone. The run log in the
workspace holds the tracebacks."""

import asyncio
import contextlib
import json
import logging
import re
import sqlite3
import tempfile
import zipfile
from collections import Counter
from collections.abc import Mapping, Sequence
from decimal import Decimal, InvalidOperation
from importlib import metadata
from pathlib import Path
from typing import Annotated, Any, NoReturn

import typer
from pydantic import ValidationError
from pypdfium2.raw import (  # pyright: ignore[reportMissingTypeStubs]
    FPDF_ERR_PASSWORD,  # pyright: ignore[reportUnknownVariableType]
)
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table
from ruamel.yaml.error import YAMLError
from typer.core import TyperGroup

from autograder2 import prompts
from autograder2.config import (
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
from autograder2.eval.eval import GoldError, evaluate, markdown, table
from autograder2.grade.audit import MAX_CASES
from autograder2.grade.ingest import IngestSummary, ingest, pdfs
from autograder2.grade.read import problem_of
from autograder2.grade.second import vendor_problem
from autograder2.grade.second_grade import PROFILES
from autograder2.grade.second_grade import ROLE as SECOND_GRADE
from autograder2.imaging import render
from autograder2.llm.client import (
    FatalError,
    Gateway,
    ResponseValidationError,
    RetryableError,
    StageError,
    StopError,
)
from autograder2.llm.ledger import Ledger
from autograder2.llm.models import (
    ModelError,
    ModelInfo,
    check,
    fetch_catalog,
    recheck_providers,
)
from autograder2.orchestrator import (
    AUDIT_GROUPS,
    GRADE_CROP,
    PATCH,
    READ_REGION,
    SECOND_READS,
    STAGES,
    Cost,
    blank_sizes,
    estimate,
    grade_class,
    grade_cost,
    privacy_problems,
    uncached,
)
from autograder2.prepare.assemble import assemble
from autograder2.prepare.key import (
    KeyFileError,
    constraint_text,
    examine,
    load_key,
    number,
    page_points,
    part_hash,
    point_problems,
    save_key,
    spec_point_problems,
    write_markdown,
    yaml_reader,
)
from autograder2.prepare.parse import SpecError, missing_numbers, parse
from autograder2.prepare.parse import check as check_spec
from autograder2.prepare.rubric import write_rubric
from autograder2.prepare.selftest import NotEarned, self_test
from autograder2.prepare.solve import PartSolution, solve
from autograder2.prepare.verify import (
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
from autograder2.report.overrides import (
    OverrideError,
    load_overrides,
    load_reviews,
    record_override,
    record_overrides,
    record_review,
)
from autograder2.report.report import (
    NOTHING_TO_REVIEW,
    Entry,
    Score,
    review_entries,
    review_line,
    write_report,
)
from autograder2.schemas.flags import Flag
from autograder2.schemas.grade import Override
from autograder2.schemas.key import Key, KeyPart, PointConstraint, Policies
from autograder2.schemas.spec import Spec
from autograder2.schemas.submission import IngestRecord
from autograder2.workspace import (
    Workspace,
    keep_out_of_git,
    log_failure,
    log_note,
    read_model,
    write_model,
)

# Where `--workspace` records the workspace a command works in, for `Commands`.
WORKSPACE = "autograder2.workspace"
# A paid command's workspace, its ledger's total when it began, and whether it
# writes the run report: noted by `_paid`, said by `Commands` once it ends.
_began: list[tuple[Workspace, Decimal, bool]] = []


class Commands(TyperGroup):
    """The command group: a workspace that cannot be written to is said in one line
    that names the workspace only, since a path below it may hold a student key,
    which holds the student's name; and a paid command ends by saying what it
    spent, or, stopped, what it spent if anything."""

    def invoke(self, ctx: Any) -> Any:
        _began.clear()
        try:
            done = super().invoke(ctx)
        except PermissionError as error:
            root = ctx.meta.get(WORKSPACE)
            if root is None or not Path(str(error.filename)).is_relative_to(root):
                raise
            _fail(
                f"{root} cannot be written to (permission denied). Make it and the"
                " files in it writable, and run the same command again."
            )
        except BaseException:  # a refusal, a failure, an interrupt
            _spent(stopped=True)
            raise
        _spent(stopped=False)
        return done


def _paid(ws: Workspace, report: bool) -> None:
    """Note, unless an earlier step of the command did, what the ledger of `ws` holds
    now, so that the command ends by saying what it spent since."""
    if not _began:
        _began.append((ws, _ledger_total(ws), report))


def _ledger_total(ws: Workspace) -> Decimal:
    """What the usage ledger of `ws` has booked in all, in USD."""
    if not ws.cache_sqlite.exists():
        return Decimal(0)
    try:
        with contextlib.closing(sqlite3.connect(ws.cache_sqlite)) as db:
            booked = Ledger(db).totals_by_role().values()
    except sqlite3.Error:  # damaged: the step that opens the cache says so
        return Decimal(0)
    return sum((u.cost for u in booked), Decimal(0))


def _spent(stopped: bool) -> None:
    """Say what the paid command spent, pointing to the run report where it writes
    one; a command stopped says it only if it spent something."""
    if not _began:
        return
    ws, before, report = _began.pop()
    spent = _ledger_total(ws) - before
    if stopped and not spent:
        return
    shown = report and ws.run_report_md.exists()
    where = f" {ws.run_report_md} has the cost by role." if shown else ""
    typer.echo(f"Spent by this command: ${spent:.4f}.{where}", err=True)


# An unhandled exception prints no local variable: locals hold the API key and
# student work.
app = typer.Typer(cls=Commands, pretty_exceptions_show_locals=False)
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
    Profile | None,
    typer.Option(help="Verification profile. Overrides autograder2.toml."),
]
Roster = Annotated[Path | None, typer.Option(metavar="FILE.csv", help="Class roster.")]
DryRun = Annotated[
    bool, typer.Option("--dry-run", help="Print a cost estimate and spend nothing.")
]
WorkspaceDir = Annotated[
    Path | None,
    typer.Option(
        metavar="DIR",
        help="Workspace. Default: <blank stem>.autograder2 beside BLANK.pdf.",
    ),
]


def _found(ctx: typer.Context, workspace: Path | None) -> Path | None:
    """A workspace option's value, the workspace first kept out of git and recorded
    for `Commands`."""
    if workspace is not None and not ctx.resilient_parsing:
        ctx.meta[WORKSPACE] = workspace
        keep_out_of_git(Workspace(workspace))
    return workspace


KeyWorkspace = Annotated[
    Path,
    typer.Option(
        "--workspace",
        metavar="DIR",
        help="The assignment's workspace.",
        callback=_found,
    ),
]


def _version(value: bool) -> None:
    if value:
        typer.echo(f"autograder2 {metadata.version('autograder2')}")
        raise typer.Exit()


@app.callback()
def main(
    verbose: Annotated[
        bool, typer.Option("-v", "--verbose", help="Log debug detail.")
    ] = False,
    version: Annotated[
        bool,
        typer.Option(
            "--version",
            callback=_version,
            is_eager=True,
            help="Print the installed version and exit.",
        ),
    ] = False,
) -> None:
    """Grade student work on an assignment."""
    logging.basicConfig(
        format="%(message)s", handlers=[RichHandler(console=Console(stderr=True))]
    )
    logging.getLogger("autograder2").setLevel(
        logging.DEBUG if verbose else logging.INFO
    )


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
    ws = _workspace(blank, workspace)
    keep_out_of_git(ws)
    if dry_run:
        _estimate(ws, blank, _dry_ingest(submissions, roster), profile)
        return
    _paid(ws, report=True)
    try:  # before a key is paid for
        given = pdfs(submissions)
    except (OSError, ValueError, zipfile.BadZipFile) as error:
        _unreadable(submissions, error)
    if not given:
        _no_pdf(submissions)
    prepare(blank, total, page_points, profile, roster, False, workspace)
    _grade(ws, submissions, roster, dry_run=False)


@app.command()
def prepare(
    blank: Blank,
    total: Total = None,
    page_points: PagePoints = None,
    profile: ProfileOption = None,
    # Hidden: `run` passes them on, and prepare ignores the one and refuses the other.
    roster: Annotated[Path | None, typer.Option(hidden=True)] = None,
    dry_run: Annotated[bool, typer.Option("--dry-run", hidden=True)] = False,
    workspace: WorkspaceDir = None,
) -> None:
    """Build the key only. On a workspace that has one, --total or --page-points
    sets its points and changes nothing else, with no model call."""
    points = _constraint(total, page_points)
    ws = _workspace(blank, workspace)
    keep_out_of_git(ws)  # a workspace that exists; a new one once the blank is in
    # Refused rather than ignored: ignoring --dry-run would spend money.
    if dry_run:
        _fail(
            "prepare has no dry run. `autograder2 run BLANK.pdf SUBMISSIONS --dry-run`"
            " estimates the key and the grading, and sends nothing."
        )
    _paid(ws, report=False)
    # Opened before anything is copied, so a bad file leaves the workspace free.
    try:
        data = blank.read_bytes()
        with render.open_pdf(data) as pdf:
            pages = len(pdf)
    except Exception as error:  # pdfium reports a file it cannot open as PdfiumError
        _no_blank(blank, error)
    if not pages:
        _no_blank(blank, None)
    try:
        config = load_config()
        if profile is not None:
            config = config.model_copy(update={"profile": profile})
        api_key = openrouter_api_key()
        if ws.blank_pdf.exists() and ws.blank_pdf.read_bytes() != data:
            _damaged(ws)  # a blank cut short is not another blank
            _fail(f"{ws.root} holds the key of another blank. Use another --workspace.")
        if not ws.blank_pdf.exists():  # whole or not at all, as write_model writes
            ws.blank_pdf.parent.mkdir(parents=True, exist_ok=True)
            partial = ws.blank_pdf.with_name(ws.blank_pdf.name + ".partial")
            partial.write_bytes(data)
            partial.replace(ws.blank_pdf)
        keep_out_of_git(ws)
        if ws.config_lock_toml.exists():
            try:
                read_lock(ws.config_lock_toml)
            except ValueError:  # not TOML or not a lock: its text stays out
                _damaged(ws)
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
        if not (ws.spec_json.exists() and ws.key_yaml.exists()):  # a model is asked
            _pinned_as_set(ws, config, lock)
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


def _pinned_as_set(ws: Workspace, config: Config, lock: Lock) -> None:
    """Exit with one line, before any call, when strict privacy is on and the lock
    was pinned with it off: its models have no endpoint that retains nothing."""
    if config.strict_privacy and any(not m.providers for m in lock.models.values()):
        _fail(
            f"{ws.config_lock_toml} was pinned with strict_privacy off, and it is on"
            " now, so nothing is sent. Set strict_privacy = false in autograder2.toml,"
            " or delete the lock and run `autograder2 prepare` again to pin endpoints"
            " that retain nothing."
        )


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
            " `autograder2 key verify` tries them again.",
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
    """What stops the key's parts and points being placed on the blank of `ws`, one
    line each: a part on a page the blank does not have, or points that cannot be
    met. Without points, a blank that does not open is no problem here."""
    try:
        with render.open_pdf(ws.blank_pdf) as pdf:
            pages = len(pdf)
    except Exception:  # pdfium reports a file it cannot open as PdfiumError
        if key.points is None:
            return []
        return [f"{ws.blank_pdf} cannot be opened to count its pages"]
    placed = [
        f"{p.id} has home page {p.home_page}, and the blank has {pages} pages"
        for p in key.parts
        if p.home_page > pages
    ]
    # Without points nothing is summed: a huge criterion is key check's to name.
    return placed + (point_problems(key, pages) if key.points is not None else [])


def _points(key: Key) -> None:
    pages = page_points(key)
    typer.echo(f"Total points: {number(sum(pages.values(), Decimal(0)))}")
    for page, points in pages.items():
        typer.echo(f"  page {page}: {number(points)}")


def _does_not_load(path: Path) -> NoReturn:
    """A key file that does not load, in one line, with the line of a YAML error:
    its problems quote the file, and `autograder2 key check` lists them for whoever
    edits it."""
    try:
        yaml_reader().load(path)  # pyright: ignore[reportUnknownMemberType]
        where = "load"
    except YAMLError as error:
        mark = getattr(error, "problem_mark", None)
        where = "parse as YAML" + (f" at line {mark.line + 1}" if mark else "")
    except OSError:
        where = "load"
    _fail(f"{path} does not {where}. `autograder2 key check` lists its problems.")


GIVE_BLANK = "Give BLANK.pdf as a PDF file that opens without a password."
GIVE_SUBMISSIONS = "Give SUBMISSIONS as a folder or ZIP of the students' PDFs."


def _no_blank(blank: Path, error: Exception | None) -> NoReturn:
    """Exit with one line: `blank`, which opening raised `error` (None: it opened
    with no page), cannot be the blank assignment."""
    if error is None:
        why = "has no pages"
    elif blank.is_dir():
        why = "is a folder"
    elif not blank.exists():
        why = "does not exist"
    # pdfium's error code is its last, which may be another file's: the blank's own
    # encryption dictionary decides.
    elif (
        getattr(error, "err_code", None) == FPDF_ERR_PASSWORD
        and b"/Encrypt" in blank.read_bytes()
    ):
        why = "is encrypted"
    else:
        why = "is not a PDF, or is damaged or cut short"
    _fail(f"{blank} cannot be opened as a PDF: it {why}. {GIVE_BLANK}")


def _workspace(blank: Path, workspace: Path | None) -> Workspace:
    """`Workspace.for_blank`, or exit with one line for a blank with no name, such
    as `.`: a folder, which names no workspace beside it."""
    if not blank.name:
        _fail(f"{blank} cannot be opened as a PDF: it is a folder. {GIVE_BLANK}")
    return Workspace.for_blank(blank, workspace)


def _no_key(ws: Workspace) -> NoReturn:
    _fail(
        f"{ws.root} holds no prepared key. Run `autograder2 prepare BLANK.pdf"
        f" --workspace {ws.root}` first."
    )


def _no_class(ws: Workspace) -> NoReturn:
    _fail(
        f"{ws.root} holds no ingest of a class. Run `autograder2 grade SUBMISSIONS"
        f" --workspace {ws.root}` first."
    )


def _fail(message: str) -> NoReturn:
    typer.echo(message, err=True)
    raise typer.Exit(code=1)


def _damaged(ws: Workspace) -> None:
    """Exit naming, a line each, the stored blank, the lock and every stored JSON file
    of `ws` that does not load (cut short or emptied, by a full disk or by hand) and
    how to have it made again, if there is one. Those of students are only counted,
    since a student's key holds the student's name: the run log names them."""
    lock, damaged = ws.config_lock_toml, False
    if ws.blank_pdf.exists():
        try:
            with render.open_pdf(ws.blank_pdf):
                pass
        except Exception:  # pdfium reports a file it cannot open as PdfiumError
            damaged = True
            typer.echo(
                f"{ws.blank_pdf} is damaged (not a PDF). Delete it and run `autograder2"
                " prepare` again with the same BLANK.pdf to have it copied again.",
                err=True,
            )
    if lock.exists():
        try:
            read_lock(lock)
        except ValueError:  # not TOML, or not a lock
            damaged = True
            typer.echo(
                f"{lock} is damaged (not a lock). Delete it and run"
                " `autograder2 prepare` again to have the models pinned again from"
                " autograder2.toml.",
                err=True,
            )
    found: list[Path] = []
    for path in sorted(ws.root.rglob("*.json")):
        try:
            json.loads(path.read_bytes())
        except ValueError:
            found.append(path)
    students = [p for p in found if p.is_relative_to(ws.students_dir)]
    for path in (p for p in found if p not in students):
        again = "prepare" if path.is_relative_to(ws.root / "key") else "grade"
        typer.echo(
            f"{path} is damaged (not JSON). Delete it and run `autograder2 {again}`"
            " again to have it made again.",
            err=True,
        )
    if students:
        log_note(ws, "damaged (not JSON)", "".join(f"{p}\n" for p in students))
        typer.echo(
            f"Stored files of students that are damaged (not JSON): {len(students)}."
            f" {ws.run_log} names them. Delete them and run `autograder2 grade` again"
            " to have them made again.",
            err=True,
        )
    if damaged or found:
        raise typer.Exit(code=1)


@key_app.command("check")
def key_check(workspace: KeyWorkspace) -> None:
    """Validate key.yaml after a hand edit: schema, ids, dependencies, points."""
    ws = Workspace(workspace)
    try:
        spec = read_model(ws.spec_json, Spec)
    except FileNotFoundError:
        _no_key(ws)
    except (OSError, ValidationError) as error:
        _fail(f"{ws.spec_json} cannot be read: {error}")
    if not ws.key_yaml.exists():  # prepare stopped before the key
        _no_key(ws)
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
    """`changed` against verification.json, then what to do next: nothing without
    the file, and one line when it does not load."""
    if not ws.verification_json.exists():
        return []
    try:
        record = read_model(ws.verification_json, Verification)
    except (OSError, ValueError):  # its text may quote the file
        return [
            f"{ws.verification_json} does not load, so no part is checked for a"
            " change since it was verified."
        ]
    if found := changed(key, record):
        found.append(
            "`autograder2 regrade` grades the class again under the edited key;"
            " `autograder2 key verify` has the edit checked."
        )
    return found


@key_app.command("render")
def key_render(workspace: KeyWorkspace) -> None:
    """Rebuild solutions.md and rubric.md."""
    ws = Workspace(workspace)
    try:
        key = load_key(ws.key_yaml)
    except FileNotFoundError:
        _no_key(ws)
    except (OSError, YAMLError, ValidationError):
        _fail(
            f"{ws.key_yaml} does not load. `autograder2 key check` lists its problems."
        )
    if found := _point_problems(ws, key):
        _fail("\n".join(found))
    try:
        write_markdown(ws, key)
    except ArithmeticError:  # a point value past what Decimal can sum
        _fail(
            f"{ws.key_yaml} holds a number too large to work with."
            " `autograder2 key check` lists its problems."
        )
    typer.echo(f"Wrote {ws.solutions_md} and {ws.rubric_md.name}")


@key_app.command("verify")
def key_verify(workspace: KeyWorkspace) -> None:
    """Solve each problem again with another vendor's model and set each part's
    state. Nothing in key.yaml changes but `state`, and nobody is re-graded."""
    ws = Workspace(workspace)
    _paid(ws, report=False)
    try:
        config = load_config()
        api_key = openrouter_api_key()
    except ConfigError as error:
        _fail(str(error))
    try:
        spec = read_model(ws.spec_json, Spec)
        lock = read_lock(ws.config_lock_toml)
    except (OSError, ValueError):  # missing, or a TOML or schema error
        _damaged(ws)
        _no_key(ws)
    key, problems = examine(ws.key_yaml, spec)
    if key is None:
        _does_not_load(ws.key_yaml)
    problems += _point_problems(ws, key)
    for problem in problems:
        typer.echo(problem, err=True)
    if problems:
        _fail(f"{ws.key_yaml} has problems, so it was not verified.")
    _can_verify(ws, lock.models)
    _pinned_as_set(ws, config, lock)
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

    try:
        verified, found = asyncio.run(run())
    except StopError as error:
        _fail(str(error))
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
    """Ingest, map, read, grade and report against the existing key. Under the
    balanced and thorough profiles each problem a model graded is graded a second
    time, and an award that differs is flagged."""
    _grade(Workspace(workspace), submissions, roster, dry_run)


def _grade(
    ws: Workspace, submissions: Path, roster: Path | None, dry_run: bool
) -> None:
    """Grade the class in `submissions` against the key in `ws`: the check that the
    second reader is another vendor's model, ingest, the check that the pinned
    endpoints keep nothing, then map, read, read again where the reader was unsure,
    grade and, under the lock's profile, grade a second time, and the report. Exit
    code 0 only when nothing failed; with `dry_run`, an estimate."""
    config, api_key, spec, lock, key = _graded(ws, dry_run)
    _corrections(ws, key)  # before anything is paid for
    if dry_run:
        _estimate(ws, ws.blank_pdf, _dry_ingest(submissions, roster))
        return
    _paid(ws, report=True)
    if found := vendor_problem(lock.models):  # before anything is sent
        _fail(f"{ws.config_lock_toml}: {found}.")
    summary = _ingest(ws, submissions, roster)
    if lock.mask_names and not config.mask_names:  # once on, on for the workspace
        typer.echo(
            f"{ws.config_lock_toml} has mask_names on, which holds once on: names"
            " are blanked though autograder2.toml has it off.",
            err=True,
        )
        config = config.model_copy(update={"mask_names": True})
    lock = _gate(ws, config, lock, api_key)
    failed = _run(ws, spec, key, lock, config, api_key, summary.students, [*STAGES])
    _report(ws, key, summary)
    if summary.failed:
        typer.echo(
            f"Files not ingested: {len(summary.failed)}. {ws.run_report_md} lists"
            " them.",
            err=True,
        )
    outside, missing = len(summary.not_on_roster), len(summary.missing_from_roster)
    left = summary.ignored_files + len(summary.unsafe_members)
    if outside or missing or left:
        typer.echo(
            f"Students not on the roster: {outside}. Roster names with no"
            f" submission: {missing}. Files left out of ingest: {left}."
            f" {ws.run_report_md} lists them.",
            err=True,
        )
    if summary.duplicates:
        typer.echo(
            f"Students with more than one PDF: {len(summary.duplicates)}. The newest"
            f" of each was graded, and {ws.ingest_summary_json} names the others;"
            " remove those not meant to be graded.",
            err=True,
        )
    if failed or summary.failed:  # a student's other PDFs are only said
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
    except (OSError, ValueError):  # missing, or a TOML or schema error
        _damaged(ws)
        _no_key(ws)
    if not ws.key_yaml.exists():  # prepare stopped before the key
        _no_key(ws)
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
    lock, written with the prompt versions, the strict privacy and what grades are
    made under in this run."""
    _pinned_as_set(ws, config, lock)
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
    masked = config.mask_names or None  # off, the lock holds no line for it
    # Only a grade run sets it, so an update of the tool or a re-pin alone leaves
    # every stored grade current.
    graded = {
        "prompt": prompts.load("grade", "problem")[1],
        "model": lock.models["grader"].slug,
    }
    update = {"strict_privacy": kept, "mask_names": masked, "graded": graded}
    lock = _with_prompts(ws, lock.model_copy(update=update))
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
    """`grade_class` over `students` with `stages`, and the second grade after the
    grade stage under a lock's profile that does it, each stage's failures counted
    on stderr; each (stage, student key) that failed."""
    if "grade" in stages and lock.profile in PROFILES:
        stages = [*stages, SECOND_GRADE]
    run = grade_class(ws, spec, key, lock.models, config, api_key, students, stages)
    try:
        failed = asyncio.run(run)
    except StopError as error:  # its own line: what to do
        log_note(ws, f"stopped: {error}")
        _fail(str(error))
    except Exception as error:  # outside any one student: the cache or the ledger
        log_failure(ws, "grade", error)
        _damaged(ws)
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


def _corrections(ws: Workspace, key: Key) -> None:
    """Exit with one line naming the entry unless overrides.yaml and reviews.yaml
    load against the key."""
    try:
        load_overrides(ws.overrides_yaml, key)
        load_reviews(ws.reviews_yaml, key)
    except OverrideError as error:
        _fail(str(error))


def _report(ws: Workspace, key: Key, summary: IngestSummary) -> list[Flag]:
    """Rebuild `out/` for the ingest's students, and say where and how long the
    review list is; the open flags."""
    raised = _rebuilt(ws, key, summary)
    typer.echo(f"Wrote {ws.root / 'out'}")
    typer.echo(review_line(ws, raised))
    return raised


def _rebuilt(ws: Workspace, key: Key, summary: IngestSummary) -> list[Flag]:
    """Rebuild `out/` for the ingest's students, saying nothing, or exit with one
    line; the open flags."""
    try:
        failed = [f.file for f in summary.failed]
        return write_report(ws, key, summary.students, failed, summary)
    except OverrideError as error:  # a file changed since it was checked
        _fail(str(error))
    except Exception as error:  # written to the log: it may quote stored work
        log_failure(ws, "report", error)
        _damaged(ws)
        _fail(f"The report could not be written. The traceback is in {ws.run_log}.")


def _ingest(ws: Workspace, submissions: Path, roster: Path | None) -> IngestSummary:
    """Ingest `submissions` and store the summary, whose students every later step
    takes. No student at all is a failure."""
    try:
        summary = ingest(submissions, ws, roster)
    except (OSError, ValueError, zipfile.BadZipFile) as error:
        _unreadable(submissions, error)
    if not summary.students:  # the stored class stays as it was
        if failed := len(summary.failed):
            _fail(
                f"No student PDF could be ingested from {submissions}: {failed} could"
                " not be read."
            )
        _no_pdf(submissions)
    write_model(ws.ingest_summary_json, summary)
    return summary


def _unreadable(submissions: Path, error: Exception) -> NoReturn:
    _fail(
        f"{submissions} cannot be read as a folder or ZIP of PDFs, or the roster"
        f" cannot be read ({type(error).__name__}). {GIVE_SUBMISSIONS}"
    )


def _no_pdf(submissions: Path) -> NoReturn:
    _fail(f"No student PDF was found in {submissions}. {GIVE_SUBMISSIONS}")


def _dry_ingest(submissions: Path, roster: Path | None) -> IngestSummary:
    """`submissions` ingested into a folder deleted afterwards, so that a dry run
    leaves the workspace's stored class and its students' files as they were."""
    with tempfile.TemporaryDirectory() as folder:
        return _ingest(Workspace(Path(folder)), submissions, roster)


# The roles the estimate prices.
ESTIMATED = ("parser", "solver", CHECKER, REFEREE, *STAGES.values())


def _estimate(
    ws: Workspace, blank: Path, summary: IngestSummary, profile: Profile | None = None
) -> None:
    """Print what preparing (unless the key exists) and grading the ingested class
    will cost, under the lock's profile, else `profile` or the configured one, with
    the assumptions. No model is asked anything."""
    try:
        config = load_config()
        profile = profile or config.profile
        if ws.config_lock_toml.exists():
            lock = read_lock(ws.config_lock_toml)
            models, pinned, profile = lock.models, True, lock.profile
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
        _damaged(ws)
        _fail(f"No estimate: a stored file cannot be read ({type(error).__name__}).")
    try:
        sizes = blank_sizes(blank)
    except Exception as error:  # pdfium reports a file it cannot open as PdfiumError
        _no_blank(blank, error)
    if not sizes:
        _no_blank(blank, None)
    keyed = ws.key_yaml.exists()
    if pinned and not keyed:  # the estimate prices a solve and its verification
        _can_verify(ws, models)
    budget = config.image_budget
    costs = estimate(models, sizes, spec, keyed, summary.students, budget, profile)
    _print_estimate(costs, models, pinned, sizes, spec, keyed, summary.students, budget)
    if found := vendor_problem(models):  # the run itself would be refused
        source = ws.config_lock_toml if pinned else "autograder2.toml"
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
        f"  {'stage':<13}{'calls':>7}{'tokens in':>12}{'tokens out':>12}{'USD':>9}"
    )
    for stage, c in [*costs.items(), ("total", total)]:
        typer.echo(
            f"  {stage:<13}{c.calls:>7.0f}{c.tokens_in:>12,.0f}"
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
        "Text and output tokens per call as measured on the fixture run. Not counted:"
        " a retry, the second map call for a page found sideways, and the parse calls"
        " made again when the question numbers have a gap. Prompt caching, not counted"
        " either, costs less.",
        f"Known only once the run is made: the second reader is assumed to read"
        f" again {SECOND_READS:.0%} of what the reader reads, and the class audit to"
        f" form {number(Decimal(str(AUDIT_GROUPS)))} groups for each part a model"
        f" grades, asked once per {MAX_CASES} students.",
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
    workspace: KeyWorkspace,
    dry_run: Annotated[
        bool,
        typer.Option(
            "--dry-run",
            help="Print how many grade requests the call cache would not answer,"
            " and send nothing.",
        ),
    ] = False,
) -> None:
    """Grade the latest ingest again from its stored readings, and a second time
    under the lock's profile, then audit and report: no ingest, map or read. The
    call cache answers each grade request that did not change, so only what a key
    edit changed is paid for."""
    ws = Workspace(workspace)
    config, api_key, spec, lock, key = _graded(ws, dry_run)
    try:
        summary = read_model(ws.ingest_summary_json, IngestSummary)
    except (OSError, ValueError):
        _damaged(ws)
        _no_class(ws)
    students = summary.students
    _corrections(ws, key)
    if dry_run:
        if (grader := lock.models.get("grader")) is None:
            _fail(f"{ws.config_lock_toml}: no model is pinned for grader.")
        count = asyncio.run(uncached(ws, key, grader, students, lock.profile))
        typer.echo(f"Grade requests the call cache would not answer: {count}.")
        usd = grade_cost(grader, spec, count)
        typer.echo(
            f"Their estimate, with nothing sent: ${usd:.4f}, each as `run --dry-run`"
            " prices a grade call: the text and output tokens measured on the"
            f" fixture run, {GRADE_CROP} tokens a crop, and"
            f" {grader.slug} at {number(grader.prompt_price * 10**6)} in and"
            f" {number(grader.completion_price * 10**6)} out per million tokens. No"
            " retry; prompt caching, not counted, costs less."
        )
        return
    _paid(ws, report=True)
    # No map or read: the images it sends are as the grade run masked them.
    config = config.model_copy(update={"mask_names": bool(lock.mask_names)})
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
    """Rebuild the files in out/ from stored readings, grades and corrections. No
    model call."""
    ws = Workspace(workspace)
    key, summary = _stored(ws)
    _corrections(ws, key)
    _report(ws, key, summary)


def _stored(ws: Workspace) -> tuple[Key, IngestSummary]:
    """The key and the latest ingest, or exit with one line."""
    try:
        key = load_key(ws.key_yaml)
    except OSError:
        _no_key(ws)
    except (YAMLError, ValidationError):
        _does_not_load(ws.key_yaml)
    try:
        summary = read_model(ws.ingest_summary_json, IngestSummary)
    except (OSError, ValidationError):
        _damaged(ws)
        _no_class(ws)
    if found := _point_problems(ws, key):
        _fail("\n".join(found))
    return key, summary


# `autograder2 review`'s actions, and its last line.
ACTIONS = ("accept", "correct", "skip", "quit")
REVIEWED = (
    "Reviewed: {reviewed}, {corrected} of them with corrections. Skipped: {skipped}."
    " Still open: {left}."
)


@app.command()
def review(
    workspace: KeyWorkspace,
    listing: Annotated[
        bool,
        typer.Option("--list", help="Print each open entry in one line; ask nothing."),
    ] = False,
) -> None:
    """Step through the open entries of the review list, in its order, and accept,
    correct or skip each. Each action is stored as it is taken, and out/ is rebuilt
    at the end. The one command that shows what a student or a model wrote, on
    stdout. No model call."""
    ws = Workspace(workspace)
    key, summary = _stored(ws)
    _corrections(ws, key)
    found = [e for e in _entries(ws, key, summary) if not e.closed]
    if not found:
        if not listing:  # out/ current, even after a session killed before it
            _rebuilt(ws, key, summary)
        typer.echo(NOTHING_TO_REVIEW)
        return
    if listing:
        for entry in found:
            typer.echo(entry.line)
        return
    done = Counter[str]()
    try:
        for entry in found:
            if (action := _take(ws, key, summary, entry)) == "quit":
                break
            done[action] += 1
    # The end of input, or an interrupt at a prompt or anywhere else: what is
    # stored stays, as on quit.
    except (typer.Abort, KeyboardInterrupt):
        typer.echo()
    except OverrideError as error:
        _fail(str(error))
    left = len({(f.student_key, f.part_id) for f in _report(ws, key, summary)})
    reviewed, corrected = done["accept"] + done["correct"], done["correct"]
    typer.echo(
        REVIEWED.format(
            reviewed=reviewed, corrected=corrected, skipped=done["skip"], left=left
        )
    )


def _entries(ws: Workspace, key: Key, summary: IngestSummary) -> list[Entry]:
    """The review list's entries for the ingest's students, or exit with one line."""
    try:
        return review_entries(ws, key, summary.students)
    except Exception as error:  # written to the log: it may quote stored work
        log_failure(ws, "review", error)
        _damaged(ws)
        _fail(f"The review list could not be made. The traceback is in {ws.run_log}.")


def _take(ws: Workspace, key: Key, summary: IngestSummary, entry: Entry) -> str:
    """Show `entry` and take one action on it, stored at once: the action taken,
    "accept" for a correction that changed nothing."""
    typer.echo()
    for line in entry.shown:
        typer.echo(line)
    offered = ACTIONS if entry.score else tuple(a for a in ACTIONS if a != "correct")
    asked = f"{', '.join(offered[:-1])} or {offered[-1]}"
    while (action := _answer(asked.capitalize()).lower()) not in offered:
        typer.echo(f"Type {asked}.")
    if action == "correct" and entry.score:
        if _correct(ws, key, entry.score, entry.review.student or ""):
            place = (entry.review.student, entry.review.part)
            [entry] = [  # as it stands with the corrections
                e
                for e in _entries(ws, key, summary)
                if (e.review.student, e.review.part) == place
            ]
        else:
            action = "accept"
    if action in ("accept", "correct"):
        record_review(ws.reviews_yaml, key, entry.review)
    return action


def _correct(ws: Workspace, key: Key, score: Score, student: str) -> bool:
    """Ask the points of each criterion of `score`'s part, return keeping the award
    shown, and a note; record those typed as `autograder2 override` does, the note on
    the first, in one write. Whether any was typed. A part withheld shows no award
    to keep."""
    shown = (
        {c.id: c.awarded for c in score.record.grade.criteria}
        if score.record
        else {c: o.points for c, o in score.overrides.items()}
    )
    typed: dict[str, Decimal] = {}
    for c in score.part.criteria:
        full, award = number(c.points), shown.get(c.id)
        keeps = "" if award is None else f", return keeps {number(award)}"
        while True:
            text = _answer(f"{c.id}, from 0 to {full}{keeps}")
            if not text and award is not None:
                break
            if (value := _award(text, c.points)) is not None:
                typed[c.id] = value
                break
            keep = "" if award is None else ", or press return to keep the award"
            typer.echo(f"Type a number from 0 to {full}{keep}.")
    if not typed:
        return False
    note = _answer("A note for the student, or return for none")
    new = [
        _correction(key, score.part, student, criterion, value, "" if n else note)
        for n, (criterion, value) in enumerate(typed.items())
    ]
    record_overrides(ws.overrides_yaml, key, new)
    return True


def _answer(prompt: str) -> str:
    return typer.prompt(prompt, default="", show_default=False).strip()


def _award(text: str, full: Decimal) -> Decimal | None:
    """`text` as points from 0 to `full`, or None: digits with an optional
    fraction, never an exponent, a sign, a comma, nan or inf."""
    text = text.strip()
    if not re.fullmatch(r"[0-9]+(\.[0-9]+)?", text):
        return None
    value = Decimal(text)
    return value if value <= full else None


def _correction(
    key: Key, part: KeyPart, student: str, criterion: str, points: Decimal, note: str
) -> Override:
    """The correction `autograder2 override` records: under the part's hash now, and
    an empty note none."""
    return Override(
        student=student,
        criterion=criterion,
        points=points,
        part_hash=part_hash(part, key.policies),
        note=note or None,
    )


# Unknown options are taken as arguments, so that POINTS below zero reaches the
# command's own check.
@app.command(context_settings={"ignore_unknown_options": True})
def override(
    ctx: typer.Context,
    student: Annotated[str, typer.Argument(help="The student key.")],
    criterion: Annotated[str, typer.Argument(help="A criterion id, such as Q7.b.c1.")],
    workspace: KeyWorkspace,
    points: Annotated[
        str | None,
        typer.Argument(help="From 0 to the criterion's full points, in its points."),
    ] = None,
    note: Annotated[
        str | None, typer.Option(help="A note for the student's feedback file.")
    ] = None,
    remove: Annotated[
        bool, typer.Option("--remove", help="Delete the correction instead.")
    ] = False,
) -> None:
    """Record a correction in overrides.yaml, in place of an earlier one for the
    student and criterion, then rebuild out/. No model call."""
    if points is not None and points.startswith("--"):  # a mistyped option
        ctx.fail(f"No such option: {points}")
    ws = Workspace(workspace)
    key, summary = _stored(ws)
    if remove:
        if points is not None or note is not None:
            _fail("--remove takes no POINTS and no --note.")
        # Whatever the entry names: a correction for a criterion the key lost, or a
        # student the class lost, can always be taken out.
        try:
            record_override(ws.overrides_yaml, key, student, criterion, None)
        except OverrideError as error:
            _fail(str(error))
        typer.echo(f"Removed from {ws.overrides_yaml}")
        _corrections(ws, key)
        _report(ws, key, summary)
        return
    _corrections(ws, key)
    if student not in [s.student_key for s in summary.students]:
        _fail(
            "No student of the latest ingest has that student key. Each key is the"
            f" name of a student's folder in {ws.students_dir}."
        )
    parts = {c.id: (p, c) for p in key.parts for c in p.criteria}
    if criterion not in parts:
        _fail("The key has no criterion with that id.")
    part, found = parts[criterion]
    if (value := _award(points or "", found.points)) is None:
        _fail(
            f"POINTS is a number from 0 to {number(found.points)}, the full"
            f" points of {criterion}."
        )
    new = _correction(key, part, student, criterion, value, note or "")
    try:
        record_override(ws.overrides_yaml, key, student, criterion, new)
    except OverrideError as error:
        _fail(str(error))
    typer.echo(f"Recorded in {ws.overrides_yaml}")
    _report(ws, key, summary)


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
        _damaged(ws)
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
            callback=_found,
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
        try:
            pinned = read_lock(lock).models
        except ValueError:  # not TOML, or not a lock
            _damaged(Workspace(lock.parent))
            raise
        rechecked = recheck_providers(pinned, api_key, strict_privacy=strict)
        found += [
            f"{lock}: {p}" for each in asyncio.run(rechecked).values() for p in each
        ]
    for problem in found:
        typer.echo(problem)
    if found:
        raise typer.Exit(code=1)
