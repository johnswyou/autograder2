"""The instructor's corrections (design 11) and reviews (design 12): `overrides.yaml`
and `reviews.yaml` in the workspace root, lists of `Override` and `Review` entries
that may be edited by hand. `load_overrides` and `load_reviews` refuse a file that
does not parse or names what does not exist; `record_override` sets or removes one
correction, and `record_review` sets one review, keeping the rest of the file as it
is, every comment included."""

# ruamel.yaml leaves the parameters of many of its methods unannotated.
# pyright: reportUnknownMemberType=false

import io
from collections.abc import Callable, Collection
from pathlib import Path
from typing import Any

from pydantic import TypeAdapter, ValidationError
from ruamel.yaml.comments import CommentedSeq
from ruamel.yaml.error import YAMLError

from redpen.prepare.key import number, part_hash, yaml_reader, yaml_writer
from redpen.schemas.common import Schema
from redpen.schemas.grade import Override, Review
from redpen.schemas.key import Key

_ENTRIES = TypeAdapter(list[Override])
_REVIEWS = TypeAdapter(list[Review])


class OverrideError(Exception):
    """`overrides.yaml` or `reviews.yaml` cannot be used, or changed as asked: one
    line, naming the entry and never quoting the file."""


def load_overrides(path: Path, key: Key, students: Collection[str]) -> list[Override]:
    """The corrections in `path`, none when there is no file. `OverrideError` when it
    does not parse; when an entry names a student key not in `students` or a
    criterion not in `key`, or repeats another's student and criterion; or when its
    points are above its criterion's full points under the part's current version
    (a stale correction keeps the points it had)."""
    return _checked(path, _text(path), key, students)


def load_reviews(path: Path, key: Key, students: Collection[str]) -> list[Review]:
    """The reviews in `path`, none when there is no file. `OverrideError` when it
    does not parse, or when an entry names a student key not in `students`, a part
    not in `key`, or neither, or repeats another's student and part."""
    return _reviewed(path, _text(path), key, students)


def _text(path: Path) -> str:
    """The text of `path`, "" when there is no file."""
    if not path.exists():
        return ""
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, ValueError) as error:
        raise OverrideError(
            f"{path} cannot be read ({type(error).__name__})."
        ) from None


def _parsed[E: Schema](
    path: Path, text: str, entries: TypeAdapter[list[E]], kind: str, holds: str
) -> list[E]:
    """`text`, the file `path`, as its entries, each a `kind` that `holds` what it
    says."""
    try:
        return entries.validate_python(yaml_reader().load(text) or [])
    except YAMLError as error:  # its message quotes the file
        mark = getattr(error, "problem_mark", None)
        line = f" at line {mark.line + 1}" if mark else ""
        raise OverrideError(f"{path} does not parse as YAML{line}.") from None
    except ValidationError as error:  # so does this one
        at = error.errors()[0]["loc"][:1]
        what = (
            f", entry {at[0] + 1} is not a {kind}"
            if at and isinstance(at[0], int)
            else f" is not a list of {kind}s"
        )
        raise OverrideError(f"{path}{what}: each has {holds}.") from None


def _checked(
    path: Path, text: str, key: Key, students: Collection[str] | None
) -> list[Override]:
    """`load_overrides` on `text`; with `students` None, no entry's student or
    criterion is checked against the class and the key."""
    found = _parsed(
        path,
        text,
        _ENTRIES,
        "correction",
        "a student key, a criterion id, points, a part_hash and an optional note",
    )
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
    _record(
        path,
        lambda text: _checked(path, text, key, students),
        lambda o: (o.student, o.criterion) == (student, criterion),
        new,
    )


def record_review(path: Path, key: Key, students: Collection[str], new: Review) -> None:
    """Make `new` the review of its entry in `path`, in place of an earlier one,
    keeping every comment and every other entry as it is, and read back first as
    `record_override` is."""
    _record(
        path,
        lambda text: _reviewed(path, text, key, students),
        lambda r: (r.student, r.part) == (new.student, new.part),
        new,
    )


def _reviewed(
    path: Path, text: str, key: Key, students: Collection[str]
) -> list[Review]:
    """`load_reviews` on `text`."""
    found = _parsed(
        path,
        text,
        _REVIEWS,
        "review",
        "a digest and, as its entry has them, a student key, a part id and a part_hash",
    )
    parts = {p.id for p in key.parts}
    first: dict[tuple[str | None, str | None], int] = {}
    for n, r in enumerate(found, 1):
        where = f"{path}, entry {n}"
        if r.student is not None and r.student not in students:
            raise OverrideError(
                f"{where}: no student of the latest ingest has its key."
            )
        if r.part is not None and r.part not in parts:
            raise OverrideError(f"{where}: the key has no part {r.part}.")
        if r.student is None and r.part is None:
            raise OverrideError(f"{where}: it names no student and no part.")
        if (earlier := first.setdefault((r.student, r.part), n)) != n:
            raise OverrideError(f"{where}: entry {earlier} reviews the same entry.")
    return found


def _record[E: Schema](
    path: Path,
    check: Callable[[str], list[E]],
    same: Callable[[E], bool],
    new: E | None,
) -> None:
    """Make `new` the entry of `path` that is the `same`, in place of an earlier
    one, or remove that one when `new` is None, as `record_override` says; `check`
    loads a text of the file."""
    text = _text(path)
    before = check(text)
    at = next((i for i, o in enumerate(before) if same(o)), None)
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
            if check(text) == expected:
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
