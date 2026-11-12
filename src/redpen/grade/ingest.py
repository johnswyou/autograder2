"""Ingest (design 7.1): a folder or ZIP of PDFs, one per student, becomes each
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
import zlib
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path, PurePosixPath, PureWindowsPath

import pypdfium2 as pdfium  # pyright: ignore[reportMissingTypeStubs]

from redpen.imaging.render import Sidebar, SourceKind, classkick_sidebars, render_page
from redpen.schemas.common import Schema
from redpen.schemas.submission import IngestPage, IngestRecord
from redpen.workspace import Workspace, read_model, write_model


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
    failed: list[Failure]  # PDFs that could not be read, and why


@dataclass(frozen=True)
class _Submission:
    file: str  # path inside the folder or ZIP, with "/"
    data: bytes
    modified: tuple[float, ...]
    pdf: pdfium.PdfDocument
    kind: SourceKind
    name: str


def ingest(
    source: Path, workspace: Workspace, roster: Path | None = None
) -> IngestSummary:
    """Ingest every PDF in the folder or ZIP `source`, one student each.

    A student's name is the sidebar's `Student:` line on a Classkick export, else
    the file name after its last " - ", else the file stem. With a roster CSV
    (names in the first column), a name that matches a roster name ignoring case
    and spacing takes the roster's spelling. Of two files for one student the most
    recently modified is used. A student whose `source.pdf` is unchanged and whose
    files are all there is not rendered again.
    """
    files, ignored, unsafe, failed = _read(source)
    spelling: dict[str, str] = {}
    if roster is not None:
        with roster.open(newline="", encoding="utf-8-sig") as rows:
            names = [row[0].strip() for row in csv.reader(rows) if row]
        spelling = {_fold(name): name for name in names if name}

    groups: dict[str, list[_Submission]] = {}
    for file, data, modified in files:
        try:
            pdf = pdfium.PdfDocument(data)
            sidebars = classkick_sidebars(pdf)
        except pdfium.PdfiumError as error:
            failed.append(Failure(file=file, reason=str(error)))
            continue
        if len(pdf) == 0:
            failed.append(Failure(file=file, reason="The PDF has no pages."))
            continue
        name = _name(file, sidebars)
        name = spelling.get(_fold(name), name)
        kind: SourceKind = "classkick" if sidebars else "other"
        submission = _Submission(file, data, modified, pdf, kind, name)
        groups.setdefault(_fold(name), []).append(submission)

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

    keys = _keys([s.name for s in chosen])
    students: list[IngestRecord] = []
    for submission in sorted(chosen, key=lambda s: keys[s.name]):
        try:
            students.append(_write(workspace, keys[submission.name], submission))
        except pdfium.PdfiumError as error:
            failed.append(Failure(file=submission.file, reason=str(error)))

    return IngestSummary(
        students=students,
        duplicates=duplicates,
        not_on_roster=sorted(
            s.name for s in chosen if roster and _fold(s.name) not in spelling
        ),
        missing_from_roster=sorted(n for f, n in spelling.items() if f not in groups),
        ignored_files=ignored,
        unsafe_members=unsafe,
        failed=failed,
    )


# A PDF's path inside the folder or ZIP (with "/"), its bytes, and when it was last
# modified: an mtime from a folder, the member's date fields from a ZIP. They are
# only compared within one source, so a ZIP's zeroed dates need no conversion.
type _File = tuple[str, bytes, tuple[float, ...]]

# What reading one ZIP member can raise: encrypted, corrupt, truncated or unsupported.
_UNREADABLE = (
    RuntimeError,
    NotImplementedError,
    EOFError,
    zipfile.BadZipFile,
    zlib.error,
)


def _read(source: Path) -> tuple[list[_File], int, list[str], list[Failure]]:
    """Each PDF in `source`; the number of other files; the ZIP members skipped for
    their names; and the ZIP members that could not be read."""
    found: list[_File] = []
    ignored = 0
    unsafe: list[str] = []
    failed: list[Failure] = []
    if source.is_dir():
        for path in sorted(p for p in source.rglob("*") if p.is_file()):
            name = path.relative_to(source).as_posix()
            if _is_pdf(name):
                found.append((name, path.read_bytes(), (path.stat().st_mtime,)))
            else:
                ignored += 1
        return found, ignored, unsafe, failed
    with zipfile.ZipFile(source) as archive:
        for member in archive.infolist():
            name = member.filename.replace("\\", "/")
            if member.is_dir():
                continue
            if (
                name.startswith("/")
                or PureWindowsPath(member.filename).drive
                or ".." in name.split("/")
            ):
                unsafe.append(member.filename)
            elif not _is_pdf(name):
                ignored += 1
            else:
                try:
                    found.append((name, archive.read(member), member.date_time))
                except _UNREADABLE as error:
                    failed.append(Failure(file=name, reason=str(error)))
    return sorted(found), ignored, unsafe, failed


def _is_pdf(name: str) -> bool:
    """A PDF that is not hidden, such as macOS's `__MACOSX/._<name>.pdf` copies."""
    hidden = any(p.startswith(".") or p == "__MACOSX" for p in name.split("/"))
    return name.lower().endswith(".pdf") and not hidden


def _name(file: str, sidebars: list[Sidebar] | None) -> str:
    for sidebar in sidebars or []:
        if sidebar.student_name:
            return sidebar.student_name
    stem = PurePosixPath(file).stem.strip()
    return stem.rpartition(" - ")[2].strip() or stem


def _fold(name: str) -> str:
    """`name` with case and spacing ignored."""
    return "".join(name.split()).casefold()


def _keys(names: list[str]) -> dict[str, str]:
    """A directory name for each student name, matching the StudentKey pattern. Two
    names that would share one get a short suffix from a hash of the name."""
    keys: dict[str, str] = {}
    taken: set[str] = set()  # casefolded, for case-insensitive file systems
    for name in sorted(names):
        plain = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
        base = re.sub(r"[^A-Za-z0-9-]+", "_", plain)[:60].strip("_-") or "student"
        key, digest, length = base, sha256(name.encode()).hexdigest(), 6
        while key.casefold() in taken:
            key, length = f"{base}-{digest[:length]}", length + 1
        taken.add(key.casefold())
        keys[name] = key
    return keys


def _write(workspace: Workspace, key: str, submission: _Submission) -> IngestRecord:
    record_path = workspace.ingest_json(key)
    source = workspace.source_pdf(key)
    if (
        record_path.is_file()
        and source.is_file()
        and source.read_bytes() == submission.data
    ):
        record = read_model(record_path, IngestRecord)
        images = [workspace.student_page(key, p.number) for p in record.pages]
        if record.student_name == submission.name and all(i.is_file() for i in images):
            return record

    # Everything in the directory derives from source.pdf, so none of it is kept, and
    # the record is written last, so a run cut short is redone in full next time.
    shutil.rmtree(workspace.student_dir(key), ignore_errors=True)
    source.parent.mkdir(parents=True)
    source.write_bytes(submission.data)
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
