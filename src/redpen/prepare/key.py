"""The key file (design 6.6): `key.yaml` read and written, checked after a hand
edit, hashed, and rendered as `solutions.md` and `rubric.md`. `key.yaml` is the only
store of answers, criteria and points. This module imports no stage, so that the
grade and report stages can load the key without the prepare stages or pdfium."""

# ruamel.yaml leaves the parameters of many of its methods unannotated. The rest of
# strict mode stays on.
# pyright: reportUnknownMemberType=false

import hashlib
import io
import json
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from decimal import ROUND_HALF_UP, Decimal, Inexact, localcontext
from pathlib import Path
from typing import Any, cast, get_origin

from pydantic import ValidationError
from ruamel.yaml import YAML
from ruamel.yaml.constructor import SafeConstructor
from ruamel.yaml.error import YAMLError
from ruamel.yaml.nodes import ScalarNode
from ruamel.yaml.representer import RoundTripRepresenter
from ruamel.yaml.scalarstring import DoubleQuotedScalarString, LiteralScalarString

from redpen.schemas.key import Answer, Key, KeyPart, PointConstraint, Policies
from redpen.schemas.spec import Part, Spec
from redpen.workspace import Workspace

# --- key.yaml ----------------------------------------------------------------------


class _Constructor(SafeConstructor):
    pass


def _float_text(constructor: SafeConstructor, node: ScalarNode) -> Any:
    return cast(str, constructor.construct_scalar(node))


# A YAML float is read as its text, so a point value such as 0.1 never passes
# through float: Pydantic makes a Decimal of the text, or a float where one is due.
_Constructor.add_constructor("tag:yaml.org,2002:float", _float_text)


class _Representer(RoundTripRepresenter):
    def ignore_aliases(self, data: Any) -> bool:
        # Criteria share one Decimal object: an anchor would let one hand edit
        # change the points of them all.
        return isinstance(data, Decimal) or super().ignore_aliases(data)


def _decimal(representer: RoundTripRepresenter, value: Decimal) -> Any:
    text = number(value)
    tag = "float" if "." in text else "int"
    return representer.represent_scalar(f"tag:yaml.org,2002:{tag}", text)


_Representer.add_representer(Decimal, _decimal)


def _writer() -> YAML:
    yaml = YAML()
    yaml.Representer = _Representer
    yaml.indent(mapping=2, sequence=4, offset=2)
    return yaml


class KeyFileError(Exception):
    """`key.yaml` would not read back as the key it was written from."""


def _reader() -> YAML:
    yaml = YAML(typ="safe", pure=True)
    yaml.Constructor = _Constructor
    return yaml


def load_key(path: Path) -> Key:
    """The key in `path`. Raises `OSError`, ruamel's `YAMLError`, or Pydantic's
    `ValidationError`."""
    data = _reader().load(path)
    _fill_answers(data)
    return Key.model_validate(data)


def _fill_answers(data: Any) -> None:
    """Give each part's answer in `data` every field: `key.yaml` leaves out those an
    answer does not use, as design 6.3 shows, and `Answer` requires them all."""
    parts = cast(dict[str, Any], data).get("parts") if isinstance(data, dict) else None
    for part in cast(list[Any], parts) if isinstance(parts, list) else []:
        answer = (
            cast(dict[str, Any], part).get("answer") if isinstance(part, dict) else None
        )
        if isinstance(answer, dict):
            absent: dict[str, Any] = {
                name: [] if get_origin(field.annotation) is list else None
                for name, field in Answer.model_fields.items()
            }
            part["answer"] = absent | cast(dict[str, Any], answer)


def save_key(path: Path, key: Key) -> None:
    """Write `key` to `path` in one step. Where `path` already holds a key, the new
    values are written into it, so that its comments and key order survive. What is
    written is read back first: if it is not `key`, `KeyFileError` is raised and
    `path` is left as it is."""
    yaml = _writer()
    data = _yaml_data(key.model_dump(exclude={"points"}))
    if key.points is not None:  # left out when unset, as in a key from before it
        data["points"] = key.points.model_dump(exclude_none=True)
    for part in data["parts"]:
        part["answer"] = {
            k: v for k, v in part["answer"].items() if v not in (None, [])
        }
    if path.exists():
        data = _merge(yaml.load(path), data)
    buffer = io.StringIO()
    yaml.dump(data, buffer)
    try:
        written = _reader().load(buffer.getvalue())
        _fill_answers(written)
        back = Key.model_validate(written)
    except (YAMLError, ValidationError):
        back = None
    if back != key:
        raise KeyFileError(
            f"{path} would not read back as the key, so it is left as it is"
        )
    partial = path.with_name(path.name + ".partial")
    partial.parent.mkdir(parents=True, exist_ok=True)
    partial.write_text(buffer.getvalue(), encoding="utf-8")
    partial.replace(path)


def _yaml_data(value: Any) -> Any:
    """`value` with each multi-line string that a literal block can hold marked for
    one, and each string a block cannot hold marked for double quotes, in which
    YAML escapes what it must."""
    if isinstance(value, dict):
        return {k: _yaml_data(v) for k, v in cast(dict[str, Any], value).items()}
    if isinstance(value, list):
        return [_yaml_data(v) for v in cast(list[Any], value)]
    if isinstance(value, str) and not _block_safe(value):
        # Typed, because assigning a plain string over a literal block keeps the block.
        return DoubleQuotedScalarString(value)
    if isinstance(value, str) and "\n" in value:
        return LiteralScalarString(value)
    return value


def _block_safe(text: str) -> bool:
    """Whether a literal block holds `text` as it is: only characters YAML prints,
    and no line break but "\\n". A block cannot escape anything."""
    return all(
        c in "\n\t"
        or " " <= c <= "~"
        or ("\xa0" <= c <= "\ufffd" and c not in "\u2028\u2029")
        or c >= "\U00010000"
        for c in text
    )


def _merge(old: Any, new: Any) -> Any:
    """`new`, written into `old` wherever both are mappings, or lists of the same
    length, and keeping `old`'s value where the two are equal, so that what `old`
    carries besides its values (comments, order, number formats) is kept."""
    if isinstance(old, dict) and isinstance(new, dict):
        target, source = cast(dict[str, Any], old), cast(dict[str, Any], new)
        for gone in [k for k in target if k not in source]:
            del target[gone]
        for k, v in source.items():
            target[k] = _merge(target[k], v) if k in target else v
        return target
    if isinstance(old, list) and isinstance(new, list):
        target_list, source_list = cast(list[Any], old), cast(list[Any], new)
        if len(target_list) == len(source_list):
            for i, v in enumerate(source_list):
                target_list[i] = _merge(target_list[i], v)
            return target_list
    # cast: the isinstance checks above leave both narrowed to partly unknown types
    return cast(Any, old) if old == new else cast(Any, new)


# --- hashes (design 6.6) -------------------------------------------------------------


def grading_content(part: KeyPart) -> str:
    """What the grader is given of `part`, as JSON: every field but `state` and
    `home_page`, each point value in its shortest form. Task 2.5 sends this text."""
    return json.dumps(part.model_dump(exclude={"state", "home_page"}), default=number)


def part_hash(part: KeyPart, policies: Policies) -> str:
    """The short hash of `part`'s grading content and the policies, so that nothing
    the grader sees changes without it. Marking a part verified or moving it to
    another page changes no hash and re-grades nobody. It does not depend on key
    order or the YAML's formatting. The parts that depend on an edited part are
    found through `depends_on`, not through this hash."""
    policy = json.dumps(policies.model_dump(), default=number)
    return _digest(f"{grading_content(part)}\n{policy}")


def key_version(key: Key) -> str:
    """The short hash of the whole key."""
    return _digest(json.dumps(key.model_dump(), sort_keys=True, default=number))


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:12]


def number(value: Decimal | float) -> str:
    """`value` in its shortest plain form: 1, 0.5, 2.5, -4. A value too large for
    Decimal arithmetic, such as 1e1000000, is given as it is."""
    try:
        return format(Decimal(str(value)).normalize(), "f")
    except ArithmeticError:
        return str(value)


# --- redpen key check ---------------------------------------------------------------


def examine(path: Path, spec: Spec) -> tuple[Key | None, list[str]]:
    """The key in `path` if it loads, and every problem `redpen key check` reports:
    the file's own (read, YAML, schema) or, once it loads, those of `check`. A
    schema error is placed by part id when it lies inside a part."""
    try:
        data = _reader().load(path)
    except (OSError, YAMLError) as error:
        return None, [str(error)]
    _fill_answers(data)
    try:
        key = Key.model_validate(data)
    except ValidationError as error:
        return None, [_located(data, e["loc"], e["msg"]) for e in error.errors()]
    return key, check(key, spec)


def _located(data: Any, loc: tuple[int | str, ...], message: str) -> str:
    """`message` after where it applies: the part's id in place of `parts.<n>`, and
    nothing at all for the whole file."""
    where = ".".join(str(x) for x in loc)
    if loc[:1] == ("parts",) and len(loc) > 1 and isinstance(loc[1], int):
        part = cast(dict[str, Any], data)["parts"][loc[1]]
        id = cast(dict[str, Any], part).get("id") if isinstance(part, dict) else None
        if isinstance(id, str) and id:
            where = ": ".join([id, ".".join(str(x) for x in loc[2:])]).rstrip(": ")
    return f"{where}: {message}" if where else message


def check(key: Key, spec: Spec) -> list[str]:
    """Every problem of a key that loads, against the spec it was made from."""
    step = key.policies.point_step
    ids = [part.id for part in key.parts]
    criteria = Counter(c.id for part in key.parts for c in part.criteria)
    problems = [
        f"part {id} appears {n} times" for id, n in Counter(ids).items() if n > 1
    ]
    problems += [
        f"criterion {id} appears {n} times" for id, n in criteria.items() if n > 1
    ]
    if step <= 0:
        problems.append(f"the point step is {number(step)}, not above zero")

    def on_step(what: str, value: Decimal) -> None:
        if step <= 0:
            return
        try:
            off = value % step != 0
        except ArithmeticError:  # too many digits for the quotient
            problems.append(
                f"{what} {number(value)}, too large to check against the point"
                f" step {number(step)}"
            )
            return
        if off:
            problems.append(
                f"{what} {number(value)}, not a multiple of the point step"
                f" {number(step)}"
            )

    for index, part in enumerate(key.parts):
        problems += [
            f"{part.id} depends on {other}, which is not an earlier part"
            for other in part.depends_on
            if other not in ids[:index]
        ]
        if not part.criteria:
            problems.append(f"{part.id} has no criteria")
        for c in part.criteria:
            if c.id.rsplit(".", 1)[0] != part.id:
                problems.append(f"criterion {c.id} is not an id of part {part.id}")
            if c.points == 0:
                problems.append(f"{c.id} is worth 0 points")
            else:
                on_step(f"{c.id} is worth", c.points)
            for level in c.partial:
                on_step(f"{c.id} has a partial level of", level.points)
                if level.points >= c.points:
                    problems.append(
                        f"{c.id} has a partial level of {number(level.points)}, not"
                        f" below its {_points(c.points)}"
                    )
    in_spec = [part.id for part in spec.parts]
    problems += [
        f"{id} is in the spec but not in the key" for id in in_spec if id not in ids
    ]
    problems += [
        f"{id} is in the key but not in the spec" for id in ids if id not in in_spec
    ]
    return problems


# --- points (design 6.4, as the instructor ruled) -----------------------------------


def weight(part: KeyPart) -> Decimal:
    """The sum of `part`'s criteria's points."""
    return sum((c.points for c in part.criteria), Decimal(0))


def allocation(key: Key) -> dict[str, Decimal]:
    """Each part's points, by part id in key order: every maximum and every score
    goes through this. With no constraint, a part is worth its weight. Under page
    maximums the parts printed on a page share its points, and under a total all
    parts share it: in proportion to their weights, in whole steps by the largest
    remainder (a tie to the earlier part). Then a part left with nothing takes one
    step from the lightest of the parts that hold the most (the later one when
    their weights are equal too). Raises ValueError for a constraint that cannot be
    met."""
    weights = {p.id: weight(p) for p in key.parts}
    if key.points is None:
        return weights
    if problems := point_problems(key):
        raise ValueError(problems[0])
    step = key.policies.point_step
    points: dict[str, Decimal] = {}
    for _, _, value, parts in _shares(key.points, key.parts):
        if parts:
            steps = _steps(int(value / step), [weights[p.id] for p in parts])
            # In its shortest form, as `number` prints it: 2, not 2.00.
            points |= {
                p.id: Decimal(number(n * step))
                for p, n in zip(parts, steps, strict=True)
            }
    return {p.id: points[p.id] for p in key.parts}


def _steps(steps: int, weights: Sequence[Decimal]) -> list[int]:
    """`steps` shared in proportion to `weights`, in whole steps, none left at 0."""
    whole = sum(weights, Decimal(0))
    with localcontext() as exact:
        exact.traps[Inexact] = True  # a share past 28 digits raises, never rounds
        cuts = [divmod(steps * w, whole) for w in weights]
    given = [int(floor) for floor, _ in cuts]
    # A stable sort: of equal remainders, the earlier part comes first.
    for i in sorted(range(len(cuts)), key=lambda i: -cuts[i][1])[: steps - sum(given)]:
        given[i] += 1
    for i in range(len(given)):
        if not given[i]:
            most = [j for j in range(len(given)) if given[j] == max(given)]
            # Of those, the lightest gives, so that no heavier part ends below it.
            donor = min(most, key=lambda j: (weights[j], -j))
            given[donor] -= 1
            given[i] += 1
    return given


def _shares(
    c: PointConstraint, parts: Sequence[KeyPart | Part]
) -> list[tuple[str, str, Decimal, list[KeyPart | Part]]]:
    """What shares points: where, the value's name, the value and its parts. That is
    the whole assignment under a total, or else each page that has a maximum or a
    part (a page past the maximums given has 0)."""
    if c.total is not None:
        return [("the assignment", "the total", c.total, list(parts))]
    given = c.pages or []
    return [
        (
            f"page {n}",
            f"page {n}'s maximum",
            given[n - 1] if n <= len(given) else Decimal(0),
            [p for p in parts if p.home_page == n],
        )
        for n in range(1, max([len(given), *(p.home_page for p in parts)]) + 1)
    ]


def point_problems(key: Key, pages: int | None = None) -> list[str]:
    """Why the key's point constraint cannot be met, one line each naming its page:
    a value below zero or off the point step, a page with points but no part, fewer
    steps than the parts that share them, or parts that weigh nothing; and, given
    the blank's number of `pages`, page maximums of another number. Nothing for no
    constraint."""
    weights = {p.id: weight(p) for p in key.parts}
    return _problems(key.points, key.policies.point_step, key.parts, pages, weights)


def spec_point_problems(c: PointConstraint, spec: Spec, pages: int) -> list[str]:
    """`point_problems` before the key exists, so that `prepare` refuses before it
    pays for a solver: the step is the policies' default, which a new key takes,
    and every part's page is in the spec. Weights come only with the criteria."""
    return _problems(c, Policies().point_step, spec.parts, pages, None)


def _problems(
    c: PointConstraint | None,
    step: Decimal,
    all_parts: Sequence[KeyPart | Part],
    pages: int | None,
    weights: Mapping[str, Decimal] | None,
) -> list[str]:
    if c is None:
        return []
    problems: list[str] = []
    if pages is not None and c.pages is not None and len(c.pages) != pages:
        problems.append(
            f"the key gives {len(c.pages)} page maximums and the blank has"
            f" {pages} pages"
        )
    for where, name, value, parts in _shares(c, all_parts):
        try:
            whole = step > 0 and value % step == 0
        except ArithmeticError:  # too many digits for the quotient
            whole = None
        if whole is None:
            problems.append(
                f"{name} is {number(value)}, too large to share in steps of"
                f" {number(step)}"
            )
        elif value < 0:
            problems.append(f"{name} is {number(value)}, below zero")
        elif not whole:
            problems.append(
                f"{name} is {number(value)}, not a multiple of the point step"
                f" {number(step)}"
            )
        elif value and not parts:
            problems.append(f"{name} is {number(value)}, but {where} has no part")
        elif value / step < len(parts):
            steps, n = int(value / step), len(parts)
            problems.append(
                f"{name} is {number(value)}: {steps} step{'s' * (steps != 1)} of"
                f" {number(step)} for {n} part{'s' * (n != 1)}, and each part"
                " needs one"
            )
        elif (
            parts
            and weights is not None
            and not sum((weights[p.id] for p in parts), Decimal(0))
        ):
            problems.append(
                f"{name} is {number(value)}, but the parts of {where} have no"
                " criterion points to share it by"
            )
    return problems


def scaled(points: Decimal, earned: Decimal, weight: Decimal, step: Decimal) -> Decimal:
    """The score of a part worth `points` whose criteria gave `earned` of their
    `weight`: `points` x `earned` / `weight`, rounded to the nearest `step`, a tie
    rounding up. Nothing earned is 0, and everything earned is `points` exactly. In
    its shortest form: 2, not 2.00."""
    steps = points * earned / (weight * step)
    return Decimal(number(steps.to_integral_value(ROUND_HALF_UP) * step))


def constraint_text(c: PointConstraint) -> str:
    """`c` in words: "a total of 34", or "page maximums of 3, 3, 4"."""
    if c.total is not None:
        return f"a total of {number(c.total)}"
    return "page maximums of " + ", ".join(number(v) for v in c.pages or [])


def scoring(c: PointConstraint, step: Decimal) -> str:
    """How a part is scored under `c`, in two sentences."""
    return (
        f"Points follow {constraint_text(c)}. A part's score is its points times the"
        " share of its criteria's points the student earned, rounded to the nearest"
        f" {number(step)}, a tie rounding up."
    )


# --- solutions.md and rubric.md -------------------------------------------------------


def page_points(key: Key) -> dict[int, Decimal]:
    """The points of each page that holds a part, by page, from the allocation."""
    points = allocation(key)
    pages: defaultdict[int, Decimal] = defaultdict(Decimal)
    for part in key.parts:
        pages[part.home_page] += points[part.id]
    return dict(sorted(pages.items()))


def write_markdown(workspace: Workspace, key: Key) -> None:
    """Render `solutions.md` and `rubric.md` from `key`, for reading and printing.
    They are never read back."""
    workspace.solutions_md.write_text(solutions_md(key), encoding="utf-8")
    workspace.rubric_md.write_text(rubric_md(key), encoding="utf-8")


def solutions_md(key: Key) -> str:
    lines = ["# Solutions", ""]
    for part in key.parts:
        lines += [
            f"## {part.id}: {part.label}, page {part.home_page}",
            "",
            part.prompt,
            "",
            f"State: {part.state}",
            "",
            "Answer:",
            "",
            *_answer(part.answer),
            "",
            "```text",
            part.solution,
            "```",
            "",
        ]
    return "\n".join(lines)


def _answer(answer: Answer) -> list[str]:
    lines: list[str] = []
    if answer.choices:
        lines.append(f"- Choice: {', '.join(answer.choices)}")
    if answer.value is not None:
        lines.append(f"- Value: {number(answer.value)} {answer.unit or ''}".rstrip())
    elif answer.unit:
        lines.append(f"- Unit: {answer.unit}")
    if answer.rel_tol is not None:
        lines.append(f"- Relative tolerance: {number(answer.rel_tol)}")
    if answer.accept_also:
        lines.append(f"- Also accepted: {', '.join(map(number, answer.accept_also))}")
    if answer.expression:
        lines.append(f"- Expression: `{answer.expression}`")
    for e in answer.elements:
        details = ", ".join(x for x in (e.kind, e.direction, e.position) if x)
        lines.append(f"- {e.label}: {details}")
    lines += [f"- {claim}" for claim in answer.must_include]
    return lines


def rubric_md(key: Key) -> str:
    pages, worth = page_points(key), allocation(key)

    def criterion(
        value: Decimal,
    ) -> str:  # its points, or under a constraint its weight
        return _points(value) if key.points is None else f"weight {number(value)}"

    lines = ["# Rubric", "", f"Total: {_points(sum(pages.values(), Decimal(0)))}", ""]
    if key.points is not None:
        lines += [
            f"{scoring(key.points, key.policies.point_step)} The weight beside each"
            " criterion and partial level counts only inside its part.",
            "",
        ]
    lines += [
        "| Page | Points |",
        "|---|---|",
        *(f"| {page} | {number(points)} |" for page, points in pages.items()),
        "",
    ]
    for part in key.parts:
        total = worth[part.id]
        lines += [
            f"## {part.id}: {part.label}, page {part.home_page}, {_points(total)}",
            "",
            part.prompt,
            "",
        ]
        for c in part.criteria:
            notes = [criterion(c.points), c.kind.replace("_", " ")]
            notes += ["carry-forward"] if c.carry_forward else []
            lines.append(f"- {c.id} ({', '.join(notes)}): {c.requirement}")
            lines += [f"  - {criterion(p.points)} when: {p.when}" for p in c.partial]
        if part.common_errors:
            lines += ["", "Common errors:", ""]
            lines += [f"- {error}" for error in part.common_errors]
        lines.append("")
    return "\n".join(lines)


def _points(points: Decimal) -> str:
    return f"{number(points)} point{'' if points == 1 else 's'}"
