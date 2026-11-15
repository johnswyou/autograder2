"""Ingest: a folder or ZIP of PDFs, one per student, becomes each
student's `source.pdf`, page images and ingest record in the workspace.

A ZIP comes from outside. Its member names never become paths: each PDF member's
bytes are written by this code to the student's directory, and a member whose name
could reach outside a directory is skipped and reported.
"""

import csv
import re
import shutil
import unicodedata
import zipfile
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path, PurePosixPath, PureWindowsPath

import pypdfium2 as pdfium  # pyright: ignore[reportMissingTypeStubs]
from pydantic import TypeAdapter, ValidationError

from autograder2.imaging.render import (
    Sidebar,
    SourceKind,
    classkick_sidebars,
    open_pdf,
    render_page,
)
from autograder2.schemas.common import Schema, StudentKey
from autograder2.schemas.submission import IngestPage, IngestRecord
from autograder2.workspace import Workspace, read_model, write_model


class Duplicate(Schema):
    student_name: str
    used: str  # the most recently modified file
    dropped: list[str]


class Failure(Schema):
    file: str
    reason: str


class IngestSummary(Schema):
    students: list[IngestRecord]
    duplicates: list[Duplicate]
    not_on_roster: list[str]
    missing_from_roster: list[str]
    ignored_files: int  # files that are not PDFs
    unsafe_members: list[str]  # ZIP members skipped for their names
    # PDFs that could not be read, and ZIP members over the size limit, and why.
    failed: list[Failure]


@dataclass(frozen=True)
class _Submission:
    file: str  # path inside the folder or ZIP, with "/"
    data: bytes
    modified: tuple[float, ...]
    pdf: pdfium.PdfDocument
    kind: SourceKind
    given: str  # the name as the submission gives it
    name: str  # the name shown: the roster's spelling when it has one


def ingest(
    source: Path, workspace: Workspace, roster: Path | None = None
) -> IngestSummary:
    """Ingest every PDF in the folder or ZIP `source`, one student each.

    A student's name is the sidebar's `Student:` line on a Classkick export, else
    the file name after its last " - ", else the file stem. With a roster CSV
    (names in the first column), a name that matches a roster name ignoring case
    and spacing takes the roster's spelling. Of two files for one student the most
    recently modified is used. The student key comes from the name as the submission
    gives it, before any roster spelling, so it is the same in every run. A student
    whose `source.pdf` is unchanged and whose files are all there is not rendered
    again.
    """
    files, ignored, unsafe, failed = _read(source)
    spelling: dict[str, str] = {}
    if roster is not None:
        with roster.open(newline="", encoding="utf-8-sig") as rows:
            names = [_nfc(row[0]).strip() for row in csv.reader(rows) if row]
        spelling = {fold(name): name for name in names if name}

    groups: dict[str, list[_Submission]] = {}
    for file, data, modified in files:
        try:
            pdf = open_pdf(data)
            sidebars = classkick_sidebars(pdf)
        except Exception as error:  # a hostile PDF fails its own student, not the run
            failed.append(_failure(file, error))
            continue
        if len(pdf) == 0:
            failed.append(Failure(file=file, reason="The PDF has no pages."))
            continue
        given = _name(file, sidebars)
        name = spelling.get(fold(given), given)
        kind: SourceKind = "classkick" if sidebars else "other"
        submission = _Submission(file, data, modified, pdf, kind, given, name)
        groups.setdefault(fold(name), []).append(submission)

    chosen: list[_Submission] = []
    duplicates: list[Duplicate] = []
    for group in groups.values():
        *older, newest = sorted(group, key=lambda s: (s.modified, s.file))
        chosen.append(newest)
        if older:
            dropped = [s.file for s in older]
            duplicates.append(
                Duplicate(student_name=newest.name, used=newest.file, dropped=dropped)
            )

    by_key: dict[str, list[_Submission]] = {}
    for submission in chosen:
        by_key.setdefault(_key(submission.given), []).append(submission)
    students: list[IngestRecord] = []
    for key, sharing in sorted(by_key.items()):
        _STUDENT_KEY.validate_python(key)  # a bad key is a bug: stop before deleting
        if len(sharing) > 1:  # writing one would delete the other's folder
            files = ", ".join(s.file for s in sharing)
            reason = f"The student key {key} is shared by {files}."
            failed.extend(Failure(file=s.file, reason=reason) for s in sharing)
            continue
        try:
            students.append(_write(workspace, key, sharing[0]))
        except Exception as error:  # a hostile PDF fails its own student, not the run
            failed.append(_failure(sharing[0].file, error))

    return IngestSummary(
        students=students,
        duplicates=duplicates,
        not_on_roster=sorted(
            s.name for s in chosen if roster and fold(s.name) not in spelling
        ),
        missing_from_roster=sorted(n for f, n in spelling.items() if f not in groups),
        ignored_files=ignored,
        unsafe_members=unsafe,
        failed=failed,
    )


def pdfs(source: Path) -> int:
    """The number of PDFs in the folder or ZIP `source`, readable or not; raises as
    `ingest` does on a path that is neither."""
    found, _, _, failed = _read(source)
    return len(found) + len(failed)


# A PDF's path inside the folder or ZIP (with "/"), its bytes, and when it was last
# modified: an mtime from a folder, the member's date fields from a ZIP. They are
# only compared within one source, so a ZIP's zeroed dates need no conversion.
type _File = tuple[str, bytes, tuple[float, ...]]

MAX_MEMBER_BYTES = 200 * 2**20  # a PDF member of a ZIP, uncompressed

# The compressions whose reads zipfile bounds: read(n) on a deflate member asks the
# decompressor for at most max(n, 4096) bytes a step, while bzip2 and LZMA members
# are decompressed with no limit (CPython's ZipExtFile._read1).
_BOUNDED = (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED)

_STUDENT_KEY = TypeAdapter(StudentKey)


def _read(source: Path) -> tuple[list[_File], int, list[str], list[Failure]]:
    """Each PDF in `source`; the number of other files; the ZIP members skipped for
    their names; and the ZIP members too large or impossible to read."""
    found: list[_File] = []
    ignored = 0
    unsafe: list[str] = []
    failed: list[Failure] = []
    if source.is_dir():
        for path in (p for p in source.rglob("*") if p.is_file()):
            name = path.relative_to(source).as_posix()
            if _is_pdf(name):
                found.append((name, path.read_bytes(), (path.stat().st_mtime,)))
            else:
                ignored += 1
        return sorted(found), ignored, unsafe, failed  # by name, as a ZIP's are
    with zipfile.ZipFile(source) as archive:
        for member in archive.infolist():
            given = member.filename.replace("\\", "/")
            parts = [p for p in given.split("/") if p not in ("", ".")]  # "./x.pdf"
            name = "/".join(parts)
            if member.is_dir():
                continue
            if (
                given.startswith("/")
                or PureWindowsPath(member.filename).drive
                or ".." in parts
            ):
                unsafe.append(member.filename)
            elif not _is_pdf(name):
                ignored += 1
            elif member.compress_type not in _BOUNDED:
                reason = "Compressed other than by deflate, which is not read here."
                failed.append(Failure(file=name, reason=reason))
            elif member.file_size > MAX_MEMBER_BYTES:
                reason = f"Over {MAX_MEMBER_BYTES // 2**20} MB uncompressed."
                failed.append(Failure(file=name, reason=reason))
            else:
                try:
                    with archive.open(member) as stream:
                        data = stream.read(member.file_size + 1)
                except Exception as error:  # encrypted, corrupt, truncated
                    failed.append(_failure(name, error))
                    continue
                if len(data) > member.file_size:
                    reason = "Holds more than the ZIP declares."
                    failed.append(Failure(file=name, reason=reason))
                else:
                    found.append((name, data, member.date_time))
    return sorted(found), ignored, unsafe, failed


def _is_pdf(name: str) -> bool:
    """A PDF that is not hidden, such as macOS's `__MACOSX/._<name>.pdf` copies."""
    hidden = any(p.startswith(".") or p == "__MACOSX" for p in name.split("/"))
    return name.lower().endswith(".pdf") and not hidden


def _name(file: str, sidebars: list[Sidebar] | None) -> str:
    for sidebar in sidebars or []:
        if sidebar.student_name:
            return _nfc(sidebar.student_name)
    stem = _nfc(PurePosixPath(file).stem).strip()
    return stem.rpartition(" - ")[2].strip() or stem


def _nfc(text: str) -> str:
    """One form for every name: macOS writes "é" in file names as "e" and an accent."""
    return unicodedata.normalize("NFC", text)


def fold(name: str) -> str:
    """`name` with case and spacing ignored."""
    return "".join(name.split()).casefold()


def _key(given: str) -> str:
    """A directory name for a student that depends on this one name only: a slug of
    it and six hex digits of its hash, matching the StudentKey pattern."""
    plain = unicodedata.normalize("NFKD", given).encode("ascii", "ignore").decode()
    slug = re.sub(r"[^A-Za-z0-9-]+", "_", plain)[:60].strip("_-") or "student"
    return f"{slug}-{sha256(given.encode()).hexdigest()[:6]}"


def _write(workspace: Workspace, key: str, submission: _Submission) -> IngestRecord:
    record_path = workspace.ingest_json(key)
    source = workspace.source_pdf(key)
    if source.is_file() and source.read_bytes() == submission.data:
        record = _kept(workspace, key)
        if record is not None:
            if record.student_name != submission.name:  # a roster now spells it
                record = record.model_copy(update={"student_name": submission.name})
                write_model(record_path, record)
            return record
        # The same submission with a stale or missing record or page: the pages are
        # redone in place and nothing else in the directory is touched.
        record_path.unlink(missing_ok=True)
    else:
        # A new or changed submission: everything in the directory derived from the
        # old one, so none of it is kept.
        if workspace.student_dir(key).exists():
            shutil.rmtree(workspace.student_dir(key))
        source.parent.mkdir(parents=True)
        source.write_bytes(submission.data)
    # The record is written last, so a run cut short is redone next time.
    pages: list[IngestPage] = []
    for number, page in enumerate(submission.pdf, 1):
        rendered = render_page(page, submission.kind)
        path = workspace.student_page(key, number)
        path.parent.mkdir(parents=True, exist_ok=True)
        rendered.image.save(path, compress_level=1)  # lossless; ~40% faster than 6
        sidebar = rendered.sidebar
        pages.append(
            IngestPage(
                number=number,
                width_px=rendered.image.width,
                height_px=rendered.image.height,
                slide_number=sidebar.slide_number if sidebar else None,
                native_ppi=rendered.native_ppi,
            )
        )
    record = IngestRecord(
        student_name=submission.name,
        student_key=key,
        source_kind=submission.kind,
        pages=pages,
    )
    write_model(record_path, record)
    return record


def _kept(workspace: Workspace, key: str) -> IngestRecord | None:
    """The student's record if it validates and every page it lists is there."""
    try:
        record = read_model(workspace.ingest_json(key), IngestRecord)
    except (OSError, ValidationError):  # missing, or written by an older schema
        return None
    pages = [workspace.student_page(key, p.number) for p in record.pages]
    return record if all(page.is_file() for page in pages) else None


def _failure(file: str, error: Exception) -> Failure:
    return Failure(file=file, reason=f"{type(error).__name__}: {error}")
