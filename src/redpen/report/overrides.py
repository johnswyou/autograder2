"""The instructor's corrections (design 11): `overrides.yaml` in the workspace root,
a list of `Override` entries that may be edited by hand. `load_overrides` refuses a
file that does not parse or names what does not exist; `record_override` sets or
removes one entry and keeps the rest of the file as it is, every comment
included."""

# ruamel.yaml leaves the parameters of many of its methods unannotated.
# pyright: reportUnknownMemberType=false

import io
from collections.abc import Collection
from pathlib import Path
from typing import Any

from pydantic import TypeAdapter, ValidationError
from ruamel.yaml.comments import CommentedSeq
from ruamel.yaml.error import YAMLError

from redpen.prepare.key import number, part_hash, yaml_reader, yaml_writer
from redpen.schemas.grade import Override
from redpen.schemas.key import Key

_ENTRIES = TypeAdapter(list[Override])


class OverrideError(Exception):
    """`overrides.yaml` cannot be used, or changed as asked: one line, naming the
    entry and never quoting the file."""


def load_overrides(path: Path, key: Key, students: Collection[str]) -> list[Override]:
    """The corrections in `path`, none when there is no file. `OverrideError` when it
    does not parse; when an entry names a student key not in `students` or a
    criterion not in `key`, or repeats another's student and criterion; or when its
    points are above its criterion's full points under the part's current version
    (a stale correction keeps the points it had)."""
    if not path.exists():
        return []
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, ValueError) as error:
        raise OverrideError(
            f"{path} cannot be read ({type(error).__name__})."
        ) from None
    return _checked(path, text, key, students)


def _checked(
    path: Path, text: str, key: Key, students: Collection[str] | None
) -> list[Override]:
    """`load_overrides` on `text`; with `students` None, no entry's student or
    criterion is checked against the class and the key."""
    try:
        found = _ENTRIES.validate_python(yaml_reader().load(text) or [])
    except YAMLError as error:  # its message quotes the file
        mark = getattr(error, "problem_mark", None)
        line = f" at line {mark.line + 1}" if mark else ""
        raise OverrideError(f"{path} does not parse as YAML{line}.") from None
    except ValidationError as error:  # so does this one
        at = error.errors()[0]["loc"][:1]
        what = (
            f", entry {at[0] + 1} is not a correction"
            if at and isinstance(at[0], int)
            else " is not a list of corrections"
        )
        raise OverrideError(
            f"{path}{what}: each has a student key, a criterion id, points, a"
            " part_hash and an optional note."
        ) from None
    parts = {c.id: (p, c) for p in key.parts for c in p.criteria}
    first: dict[tuple[str, str], int] = {}
    for n, o in enumerate(found, 1):
        where = f"{path}, entry {n}"
        if students is None:
            pass
        elif o.student not in students:
            raise OverrideError(
                f"{where}: no student of the latest ingest has its key."
            )
        elif o.criterion not in parts:
            raise OverrideError(f"{where}: the key has no criterion {o.criterion}.")
        if (earlier := first.setdefault((o.student, o.criterion), n)) != n:
            raise OverrideError(
                f"{where}: entry {earlier} corrects the same student and criterion."
            )
        part, criterion = parts.get(o.criterion, (None, None))
        if (
            part
            and criterion
            and o.part_hash == part_hash(part, key.policies)
            and o.points > criterion.points
        ):
            raise OverrideError(
                f"{where}: its points are above the {number(criterion.points)} that"
                f" {o.criterion} is worth."
            )
    return found


def record_override(
    path: Path,
    key: Key,
    students: Collection[str] | None,
    student: str,
    criterion: str,
    new: Override | None,
) -> None:
    """Make `new` the correction of `student` on `criterion` in `path`, in place of
    an earlier one, or remove that one when `new` is None, keeping every comment and
    every other entry as it is. `students` as for `_checked`: None skips the checks
    that an entry's student and criterion exist, so that a removal always works.
    What is written is read back first: unless it holds exactly the corrections
    asked for, `OverrideError` leaves `path` as it is, as when there is nothing to
    remove."""
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    before = _checked(path, text, key, students)
    mine = (student, criterion)
    at = next(
        (i for i, o in enumerate(before) if (o.student, o.criterion) == mine), None
    )
    if new is None and at is None:
        raise OverrideError("No correction is recorded for that student and criterion.")
    yaml = yaml_writer()
    entries: Any = yaml.load(text) if text.strip() else CommentedSeq()
    if new is None and at is not None:
        expected = before[:at] + before[at + 1 :]
        written = [_without(text, entries, at)]
        del entries[at]  # if lines cannot take it out, as YAML, losing its comments
    elif at is not None and new is not None:
        expected = [new if i == at else o for i, o in enumerate(before)]
        written = []
        for name, value in new.model_dump().items():
            if value is not None or name in entries[at]:  # in place: comments stay
                entries[at][name] = value
    else:
        assert new is not None
        expected = [*before, new]
        written = []
        entries.fa.set_block_style()  # an empty `[]` would put entries on one line
        entries.append(new.model_dump(exclude_none=True))
    buffer = io.StringIO()
    yaml.dump(entries, buffer)
    for text in [*written, buffer.getvalue()]:
        try:
            if _checked(path, text, key, students) == expected:
                break
        except OverrideError:
            pass
    else:
        raise OverrideError(
            f"{path} would not read back as the corrections, so it is left as it is."
        )
    partial = path.with_name(path.name + ".partial")
    partial.write_text(text, encoding="utf-8")
    partial.replace(path)


def _without(text: str, entries: Any, at: int) -> str:
    """`text` with its entry `at` taken out by lines, from the entry's first line to
    the next entry's: every blank and comment line among them stays, and so does an
    end-of-line comment, on a line of its own. `entries` is `text` loaded round
    trip."""
    lines = text.splitlines(keepends=True)
    start = entries.lc.item(at)[0]
    end = entries.lc.item(at + 1)[0] if at + 1 < len(entries) else len(lines)
    slots: list[Any] = [*entries.ca.items.get(at, []), *entries[at].ca.items.values()]
    tokens: list[Any] = []
    for slot in slots:  # a slot holds a token, a list of them, or None
        tokens += slot if isinstance(slot, list) else [slot]
    texts = [str(token.value) for token in tokens if token is not None]
    inline = [first for t in texts if (first := t.split("\n")[0]).startswith("#")]
    kept: list[str] = []
    for line in lines[start:end]:
        body = line.rstrip()
        if not body or body.lstrip().startswith("#"):
            kept.append(line)
        elif comment := next((c for c in inline if body.endswith(c)), None):
            kept.append(f"{line[: len(line) - len(line.lstrip())]}{comment}\n")
    return "".join([*lines[:start], *kept, *lines[end:]])
