"""The instructor's corrections (design 11): `overrides.yaml` in the workspace root,
a list of `Override` entries that may be edited by hand. `load_overrides` refuses a
file that does not parse or names what does not exist; `record_override` sets or
removes one entry and keeps the rest of the file, comments included."""

# ruamel.yaml leaves the parameters of many of its methods unannotated.
# pyright: reportUnknownMemberType=false

import io
from collections.abc import Collection
from pathlib import Path
from typing import Any

from pydantic import TypeAdapter, ValidationError
from ruamel.yaml.error import YAMLError

from redpen.prepare.key import number, part_hash, yaml_reader, yaml_writer
from redpen.schemas.grade import Override
from redpen.schemas.key import Key

ENTRIES = TypeAdapter(list[Override])


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
    path: Path, text: str, key: Key, students: Collection[str]
) -> list[Override]:
    try:
        found = ENTRIES.validate_python(yaml_reader().load(text) or [])
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
        if o.student not in students:
            raise OverrideError(
                f"{where}: no student of the latest ingest has its key."
            )
        if o.criterion not in parts:
            raise OverrideError(f"{where}: the key has no criterion {o.criterion}.")
        if (earlier := first.setdefault((o.student, o.criterion), n)) != n:
            raise OverrideError(
                f"{where}: entry {earlier} corrects the same student and criterion."
            )
        part, criterion = parts[o.criterion]
        if o.part_hash == part_hash(part, key.policies) and o.points > criterion.points:
            raise OverrideError(
                f"{where}: its points are above the {number(criterion.points)} that"
                f" {o.criterion} is worth."
            )
    return found


def record_override(
    path: Path,
    key: Key,
    students: Collection[str],
    student: str,
    criterion: str,
    new: Override | None,
) -> None:
    """Make `new` the correction of `student` on `criterion` in `path`, in place of
    an earlier one, or remove that one when `new` is None. The file must load. What
    is written is read back first; on `OverrideError`, as when there is nothing to
    remove, `path` is left as it is."""
    yaml = yaml_writer()
    loaded: Any = yaml.load(path.read_text(encoding="utf-8")) if path.exists() else None
    # An empty file starts afresh: `[]` would keep later entries on one line.
    entries: list[Any] = loaded if loaded else []
    at = [
        i
        for i, e in enumerate(entries)
        if (e["student"], e["criterion"]) == (student, criterion)
    ]
    if new is None and not at:
        raise OverrideError("No correction is recorded for that student and criterion.")
    if new is None:
        del entries[at[0]]
    elif at:
        entries[at[0]] = new.model_dump(exclude_none=True)
    else:
        entries.append(new.model_dump(exclude_none=True))
    buffer = io.StringIO()
    yaml.dump(entries, buffer)
    mine = [
        o
        for o in _checked(path, buffer.getvalue(), key, students)
        if (o.student, o.criterion) == (student, criterion)
    ]
    if mine != ([new] if new else []):
        raise OverrideError(
            f"{path} would not read back as the corrections, so it is left as it is."
        )
    partial = path.with_name(path.name + ".partial")
    partial.write_text(buffer.getvalue(), encoding="utf-8")
    partial.replace(path)
