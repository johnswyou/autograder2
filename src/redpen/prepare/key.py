"""The key (design 6.6): assembled from the spec and the stage outputs, kept in
`key.yaml`, checked after a hand edit, hashed, and rendered as `solutions.md` and
`rubric.md`. `key.yaml` is the only store of answers, criteria and points."""

# ruamel.yaml leaves the parameters of many of its methods unannotated. The rest of
# strict mode stays on.
# pyright: reportUnknownMemberType=false

import hashlib
import io
import json
from collections import Counter, defaultdict
from collections.abc import Mapping
from decimal import Decimal
from pathlib import Path
from typing import Any, cast, get_origin

from pydantic import ValidationError
from ruamel.yaml import YAML
from ruamel.yaml.constructor import SafeConstructor
from ruamel.yaml.error import YAMLError
from ruamel.yaml.nodes import ScalarNode
from ruamel.yaml.representer import RoundTripRepresenter
from ruamel.yaml.scalarstring import LiteralScalarString

from redpen.prepare.parse import CHOICE_TYPES
from redpen.prepare.rubric import PartRubric, ProposedCriterion
from redpen.prepare.solve import PartSolution, answer_of, depends_on
from redpen.schemas.key import Answer, Criterion, Key, KeyPart, PartialCredit, Policies
from redpen.schemas.spec import Spec
from redpen.workspace import Workspace

ONE = Decimal(1)  # every criterion's points (design 6.4, the proposal used as is)


def assemble(
    spec: Spec,
    solutions: Mapping[str, PartSolution],
    rubrics: Mapping[str, PartRubric],
) -> Key:
    """The key of `spec` under the default policies: each part's solution and
    answer, and its criteria, each worth one point with ids given in order. A choice
    part's one criterion is written here. Every part is `unverified`, since one
    solver verifies nothing."""
    policies = Policies()
    parts: list[KeyPart] = []
    for part in spec.parts:
        found = solutions[part.id]
        answer = answer_of(found)
        depends = depends_on(spec, part, found)
        if part.response_type in CHOICE_TYPES:
            only = (
                " and no other option"
                if part.response_type == "multiple_select"
                else ""
            )
            requirement = f"Chooses {' and '.join(answer.choices)}{only}."
            proposed = [
                ProposedCriterion(
                    kind="final_answer", requirement=requirement, partial=[]
                )
            ]
            errors: list[str] = []
            carries = False  # fixed options leave no result to carry forward
        else:
            proposed = rubrics[part.id].criteria
            errors = rubrics[part.id].common_errors
            carries = bool(depends)
        criteria = [
            Criterion(
                id=f"{part.id}.c{n}",
                points=ONE,
                kind=c.kind,
                requirement=c.requirement,
                partial=_levels(c.partial, ONE, policies.point_step),
                carry_forward=carries and c.kind == "final_answer",
            )
            for n, c in enumerate(proposed, 1)
        ]
        parts.append(
            KeyPart(
                id=part.id,
                label=part.label,
                home_page=part.home_page,
                response_type=part.response_type,
                depends_on=depends,
                state="unverified",
                prompt=part.prompt,
                answer=answer,
                solution=found.solution,
                criteria=criteria,
                common_errors=errors,
            )
        )
    return Key(policies=policies, parts=parts)


def _levels(
    proposed: list[PartialCredit], points: Decimal, step: Decimal
) -> list[PartialCredit]:
    """Each level rounded down to a multiple of `step`, kept if that leaves it above
    zero and below `points`."""
    kept: list[PartialCredit] = []
    for level in proposed:
        rounded = level.points // step * step
        if 0 < rounded < points:
            kept.append(level.model_copy(update={"points": rounded}))
    return kept


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
    data = _yaml_data(key.model_dump())
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
    one. Any other string is left for ruamel to quote and escape."""
    if isinstance(value, dict):
        return {k: _yaml_data(v) for k, v in cast(dict[str, Any], value).items()}
    if isinstance(value, list):
        return [_yaml_data(v) for v in cast(list[Any], value)]
    if isinstance(value, str) and "\n" in value and _block_safe(value):
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


def part_hash(part: KeyPart, policies: Policies) -> str:
    """The short hash of a key part's content and the policies it is graded under.
    It does not depend on key order or the YAML's formatting."""
    return _digest({"part": part.model_dump(), "policies": policies.model_dump()})


def key_version(key: Key) -> str:
    """The short hash of the whole key."""
    return _digest(key.model_dump())


def _digest(data: dict[str, Any]) -> str:
    text = json.dumps(data, sort_keys=True, separators=(",", ":"), default=number)
    return hashlib.sha256(text.encode()).hexdigest()[:12]


def number(value: Decimal | float) -> str:
    """`value` in its shortest plain form: 1, 0.5, 2.5, -4."""
    return format(Decimal(str(value)).normalize(), "f")


# --- redpen key check ---------------------------------------------------------------


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
    for index, part in enumerate(key.parts):
        problems += [
            f"{part.id} depends on {other}, which is not an earlier part"
            for other in part.depends_on
            if other not in ids[:index]
        ]
        for c in part.criteria:
            if c.id.rsplit(".", 1)[0] != part.id:
                problems.append(f"criterion {c.id} is not an id of part {part.id}")
            if step > 0 and c.points % step:
                problems.append(
                    f"{c.id} is worth {number(c.points)}, not a multiple of the"
                    f" point step {number(step)}"
                )
            for level in c.partial:
                if step > 0 and level.points % step:
                    problems.append(
                        f"{c.id} has a partial level of {number(level.points)}, not a"
                        f" multiple of the point step {number(step)}"
                    )
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


# --- solutions.md and rubric.md -------------------------------------------------------


def page_points(key: Key) -> dict[int, Decimal]:
    """The points of each page that holds a part, by page."""
    pages: defaultdict[int, Decimal] = defaultdict(Decimal)
    for part in key.parts:
        pages[part.home_page] += sum((c.points for c in part.criteria), Decimal(0))
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
    pages = page_points(key)
    lines = [
        "# Rubric",
        "",
        f"Total: {_points(sum(pages.values(), Decimal(0)))}",
        "",
        "| Page | Points |",
        "|---|---|",
        *(f"| {page} | {number(points)} |" for page, points in pages.items()),
        "",
    ]
    for part in key.parts:
        total = sum((c.points for c in part.criteria), Decimal(0))
        lines += [
            f"## {part.id}: {part.label}, page {part.home_page}, {_points(total)}",
            "",
            part.prompt,
            "",
        ]
        for c in part.criteria:
            notes = [_points(c.points), c.kind.replace("_", " ")]
            notes += ["carry-forward"] if c.carry_forward else []
            lines.append(f"- {c.id} ({', '.join(notes)}): {c.requirement}")
            lines += [f"  - {_points(p.points)} when: {p.when}" for p in c.partial]
        if part.common_errors:
            lines += ["", "Common errors:", ""]
            lines += [f"- {error}" for error in part.common_errors]
        lines.append("")
    return "\n".join(lines)


def _points(points: Decimal) -> str:
    return f"{number(points)} point{'' if points == 1 else 's'}"
