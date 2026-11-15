"""Ingest of a folder or ZIP of submissions (src/autograder2/grade/ingest.py)."""

import os
import shutil
import struct
import tempfile
import tracemalloc
import unicodedata
import zipfile
from hashlib import sha256
from pathlib import Path

import pypdfium2 as pdfium
import pypdfium2.raw as pdfium_c
import pytest
from PIL import Image, ImageChops
from reportlab.pdfgen.canvas import Canvas

from autograder2.grade import ingest as ingest_module
from autograder2.grade.ingest import IngestSummary, ingest
from autograder2.schemas.submission import IngestRecord
from autograder2.workspace import Workspace, read_model
from tests.fixture_truth import SUBMISSIONS, truth_of
from tests.real_data import submission_folders

OTHER = next(p for p in SUBMISSIONS if truth_of(p).source_kind == "other")
CLASSKICK = [p for p in SUBMISSIONS if truth_of(p).source_kind == "classkick"]


def folder(root: Path, files: dict[str, Path | bytes]) -> Path:
    """A folder under `root` holding each file, copied or written, under its name."""
    path = root / "submissions"
    path.mkdir(parents=True)
    for name, content in files.items():
        (path / name).parent.mkdir(parents=True, exist_ok=True)
        if isinstance(content, Path):
            shutil.copyfile(content, path / name)
        else:
            (path / name).write_bytes(content)
    return path


def zip_of(source: Path, archive: Path) -> Path:
    """A ZIP of every file under `source`, by its path relative to `source`."""
    with zipfile.ZipFile(archive, "w") as zipped:
        for path in sorted(p for p in source.rglob("*") if p.is_file()):
            zipped.write(path, path.relative_to(source).as_posix())
    return archive


def test_a_zip_and_a_folder_of_the_same_files_give_identical_results(
    tmp_path: Path,
) -> None:
    files: dict[str, Path | bytes] = {p.name: p for p in (OTHER, *CLASSKICK[:2])}
    files["notes.txt"] = b"ignored"
    source = folder(tmp_path, files)
    archive = tmp_path / "submissions.zip"
    with zipfile.ZipFile(archive, "w") as zipped:
        for path in sorted(source.iterdir()):
            zipped.write(path, path.name)
    from_folder, from_zip = Workspace(tmp_path / "a"), Workspace(tmp_path / "b")

    summary = ingest(source, from_folder)

    assert ingest(archive, from_zip) == summary
    assert len(summary.students) == 3
    assert summary.ignored_files == 1
    for record in summary.students:
        for page in record.pages:
            written = [
                ws.student_page(record.student_key, page.number).read_bytes()
                for ws in (from_folder, from_zip)
            ]
            assert written[0] == written[1]


@pytest.mark.parametrize("pdf", SUBMISSIONS, ids=lambda p: truth_of(p).layout_class)
def test_each_fixture_ingests_as_its_truth_says(
    fixture_workspace: tuple[Workspace, IngestSummary], pdf: Path
) -> None:
    workspace, summary = fixture_workspace
    truth = truth_of(pdf)
    [record] = [r for r in summary.students if r.student_name == truth.student_name]

    assert record.source_kind == truth.source_kind
    assert [p.slide_number for p in record.pages] == [
        p.slide_number for p in truth.pages
    ]
    assert [(p.width_px, p.height_px) for p in record.pages] == [
        (p.width_px, p.height_px) for p in truth.pages
    ]
    assert [p.native_ppi for p in record.pages] == pytest.approx(
        [p.native_ppi for p in truth.pages]
    )
    if truth.source_kind == "other":
        return
    for page, document_page in zip(record.pages, pdfium.PdfDocument(pdf), strict=True):
        [slide] = document_page.get_objects(filter=[pdfium_c.FPDF_PAGEOBJ_IMAGE])
        assert isinstance(slide, pdfium.PdfImage)
        rgba = slide.get_bitmap(render=True).to_pil()
        over_white = Image.alpha_composite(Image.new("RGBA", rgba.size, "white"), rgba)
        written = Image.open(workspace.student_page(record.student_key, page.number))
        assert written.mode == "RGB"
        assert written.tobytes() == over_white.convert("RGB").tobytes()
        # Independently of the compositing: opaque pixels are the raw slide's own.
        opaque = rgba.getchannel("A").point(lambda a: 255 if a == 255 else 0)
        raw = slide.get_bitmap(render=False).to_pil().convert("RGB")
        difference = ImageChops.difference(written, raw)
        assert ImageChops.multiply(difference, opaque.convert("RGB")).getbbox() is None


def test_unsafe_zip_members_are_written_nowhere_and_reported(tmp_path: Path) -> None:
    archive = tmp_path / "in" / "submissions.zip"
    archive.parent.mkdir()
    with zipfile.ZipFile(archive, "w") as zipped:
        for name in ("../evil.pdf", "/abs.pdf", f"Title - {OTHER.stem}.pdf"):
            zipped.writestr(zipfile.ZipInfo(name), OTHER.read_bytes())
        assert zipped.namelist()[:2] == ["../evil.pdf", "/abs.pdf"]
    assert not Path("/abs.pdf").exists()

    summary = ingest(archive, Workspace(tmp_path / "ws"))

    assert summary.unsafe_members == ["../evil.pdf", "/abs.pdf"]
    assert len(summary.students) == 1
    assert not Path("/abs.pdf").exists()
    assert not any(p.name in ("evil.pdf", "abs.pdf") for p in tmp_path.rglob("*"))


def test_a_zip_member_that_cannot_be_read_is_reported_and_skipped(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "submissions.zip"
    with zipfile.ZipFile(archive, "w") as zipped:
        zipped.writestr("Title - Broken.pdf", b"%PDF-1.4 to be corrupted")
        zeroed = zipfile.ZipInfo(OTHER.name, date_time=(1980, 0, 0, 0, 0, 0))
        zipped.writestr(zeroed, OTHER.read_bytes())  # as some ZIP tools write them
    archive.write_bytes(archive.read_bytes().replace(b"to be", b"TO BE"))  # bad CRC

    summary = ingest(archive, Workspace(tmp_path / "ws"))

    assert [r.student_name for r in summary.students] == [truth_of(OTHER).student_name]
    [failure] = summary.failed
    assert failure.file == "Title - Broken.pdf"
    assert "crc" in failure.reason.lower()


def declare_sizes(archive: Path, sizes: list[int]) -> None:
    """Rewrite the uncompressed size the ZIP's central directory gives each of its
    first members, in order."""
    data = bytearray(archive.read_bytes())
    at = struct.unpack("<I", data[-6:-2])[0]  # where the central directory starts
    for size in sizes:
        at = data.index(b"PK\x01\x02", at)
        data[at + 24 : at + 28] = struct.pack("<I", size)
        at += 4
    archive.write_bytes(data)


def test_an_oversized_zip_member_is_skipped_and_reported(tmp_path: Path) -> None:
    archive = tmp_path / "submissions.zip"
    with zipfile.ZipFile(archive, "w") as zipped:
        zipped.writestr("Title - Huge.pdf", b"%PDF-1.4 declared as 300 MB")
        zipped.writestr("Title - Liar.pdf", OTHER.read_bytes())
        zipped.writestr(OTHER.name, OTHER.read_bytes())
    declare_sizes(archive, [300 * 2**20, 100])  # Liar holds far more than 100 bytes

    summary = ingest(archive, Workspace(tmp_path / "ws"))

    assert [r.student_name for r in summary.students] == [truth_of(OTHER).student_name]
    reasons = {f.file: f.reason for f in summary.failed}
    assert reasons["Title - Huge.pdf"] == "Over 200 MB uncompressed."
    assert "crc" in reasons["Title - Liar.pdf"].lower()  # read stopped at 100 bytes


def corrupt(archive: Path, name: str) -> None:
    """Scramble the middle of a member's compressed data, in place."""
    with zipfile.ZipFile(archive) as zipped:
        member = zipped.getinfo(name)
    data = bytearray(archive.read_bytes())
    at = member.header_offset
    name_length, extra_length = struct.unpack("<HH", data[at + 26 : at + 30])
    start = at + 30 + name_length + extra_length
    for index in range(start + 16, start + member.compress_size - 16):
        data[index] ^= 0xFF
    archive.write_bytes(data)


@pytest.mark.parametrize(
    "compression",
    [zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED],
    ids=["stored", "deflated"],
)
def test_a_corrupt_zip_member_is_reported_and_skipped(
    tmp_path: Path, compression: int
) -> None:
    archive = tmp_path / "submissions.zip"
    with zipfile.ZipFile(archive, "w") as zipped:
        zipped.writestr("Title - Broken.pdf", OTHER.read_bytes(), compression)
        zipped.write(OTHER, OTHER.name)
    corrupt(archive, "Title - Broken.pdf")

    summary = ingest(archive, Workspace(tmp_path / "ws"))

    assert [r.student_name for r in summary.students] == [truth_of(OTHER).student_name]
    assert [f.file for f in summary.failed] == ["Title - Broken.pdf"]


def test_a_zip_member_under_dot_slash_is_ingested(tmp_path: Path) -> None:
    archive = tmp_path / "submissions.zip"
    with zipfile.ZipFile(archive, "w") as zipped:
        zipped.writestr(zipfile.ZipInfo(f"./{OTHER.name}"), OTHER.read_bytes())
        assert zipped.namelist() == [f"./{OTHER.name}"]

    summary = ingest(archive, Workspace(tmp_path / "ws"))

    assert [r.student_name for r in summary.students] == [truth_of(OTHER).student_name]
    assert summary.ignored_files == 0


def test_a_zip_and_a_folder_with_subfolders_give_identical_results(
    tmp_path: Path,
) -> None:
    # Sorted as paths, "a/" comes before "a b/"; sorted as strings, after it.
    files: dict[str, Path | bytes] = {
        "a b/Title - Quinn.pdf": OTHER,
        "a/Title - Pat.pdf": OTHER,
        "c/Title - Quinn.pdf": OTHER,
        "c/Title - Pat.pdf": OTHER,
        "a b/broken.pdf": b"not a pdf",
        "a/broken.pdf": b"not a pdf",
    }
    source = folder(tmp_path, files)
    for name in files:
        stamp = 2_000_000_000 if name.startswith("c/") else 1_500_000_000
        os.utime(source / name, (stamp, stamp))
    archive = zip_of(source, tmp_path / "submissions.zip")

    from_folder = ingest(source, Workspace(tmp_path / "folder.autograder2"))

    assert ingest(archive, Workspace(tmp_path / "zip.autograder2")) == from_folder
    assert [d.used for d in from_folder.duplicates] == [
        "c/Title - Quinn.pdf",
        "c/Title - Pat.pdf",
    ]
    assert [f.file for f in from_folder.failed] == ["a b/broken.pdf", "a/broken.pdf"]


def test_the_decomposed_and_composed_forms_of_a_name_are_one_student(
    tmp_path: Path,
) -> None:
    composed = "José Núñez"
    decomposed = unicodedata.normalize("NFD", composed)  # as macOS writes file names
    assert decomposed != composed
    roster = tmp_path / "roster.csv"
    roster.write_text(composed + "\n")
    old, new = f"Old - {decomposed}.pdf", f"New - {composed}.pdf"
    source = folder(tmp_path / "both", {old: OTHER, new: OTHER})
    os.utime(source / old, (1_500_000_000, 1_500_000_000))

    summary = ingest(source, Workspace(tmp_path / "ws"), roster)
    alone = ingest(folder(tmp_path / "nfd", {old: OTHER}), Workspace(tmp_path / "w2"))

    [record] = summary.students
    assert record.student_name == composed
    assert [(d.used, d.dropped) for d in summary.duplicates] == [(new, [old])]
    assert summary.not_on_roster == summary.missing_from_roster == []
    assert alone.students[0].student_key == record.student_key


def test_a_deflate_member_holding_more_than_it_declares_is_not_inflated(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "submissions.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zipped:
        with zipped.open("Title - Bomb.pdf", "w") as member:
            for _ in range(50):
                member.write(bytes(2**20))  # 50 MB of zeros, about 50 KB deflated
    declare_sizes(archive, [100])

    tracemalloc.start()
    try:
        summary = ingest(archive, Workspace(tmp_path / "ws"))
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()

    assert [f.file for f in summary.failed] == ["Title - Bomb.pdf"]
    assert peak < 4 * 2**20


@pytest.mark.parametrize(
    "compression", [zipfile.ZIP_BZIP2, zipfile.ZIP_LZMA], ids=["bzip2", "lzma"]
)
def test_a_member_compressed_other_than_by_deflate_is_skipped_and_reported(
    tmp_path: Path, compression: int
) -> None:
    archive = tmp_path / "submissions.zip"
    with zipfile.ZipFile(archive, "w") as zipped:
        zipped.writestr(OTHER.name, OTHER.read_bytes(), compression)

    summary = ingest(archive, Workspace(tmp_path / "ws"))

    assert summary.students == []
    [failure] = summary.failed
    assert failure.file == OTHER.name
    assert "deflate" in failure.reason


@pytest.mark.parametrize("record", ["stale", "missing"])
def test_the_same_submission_with_a_bad_record_keeps_later_work(
    tmp_path: Path, record: str
) -> None:
    source = folder(tmp_path, {OTHER.name: OTHER})
    workspace = Workspace(tmp_path / "ws")
    [first] = ingest(source, workspace).students
    grade = workspace.grade_json(first.student_key, "Q1")  # a later stage's work
    grade.parent.mkdir()
    grade.write_text("{}")
    path = workspace.ingest_json(first.student_key)
    if record == "stale":
        path.write_text('{"student_name": "old"}')
    else:
        path.unlink()

    assert ingest(source, workspace).students == [first]
    assert read_model(path, IngestRecord) == first
    assert grade.exists()


def test_a_pdf_with_a_page_of_zero_size_fails_that_student_alone(
    tmp_path: Path,
) -> None:
    pdf = pdfium.PdfDocument.new()
    page = pdf.new_page(612, 792)
    page.set_cropbox(1000, 1000, 1100, 1100)  # outside the media box: 0 x 0
    hostile = tmp_path / "hostile.pdf"
    pdf.save(hostile)
    files: dict[str, Path | bytes] = {"Title - Zero.pdf": hostile, OTHER.name: OTHER}

    summary = ingest(folder(tmp_path, files), Workspace(tmp_path / "ws"))

    assert [r.student_name for r in summary.students] == [truth_of(OTHER).student_name]
    [failure] = summary.failed
    assert failure.file == "Title - Zero.pdf"
    assert "ZeroDivisionError" in failure.reason


def test_two_students_with_one_key_are_reported_and_neither_written(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(ingest_module, "_key", lambda given: "Same-000000")
    files: dict[str, Path | bytes] = {
        "Title - Ann.pdf": OTHER,
        "Title - Bea.pdf": OTHER,
    }
    workspace = Workspace(tmp_path / "ws")

    summary = ingest(folder(tmp_path, files), workspace)

    assert summary.students == []
    assert sorted(f.file for f in summary.failed) == sorted(files)
    assert all("Same-000000" in f.reason for f in summary.failed)
    assert not workspace.student_dir("Same-000000").exists()


@pytest.mark.skipif(os.geteuid() == 0, reason="root can delete a read-only folder")
def test_a_student_whose_folder_cannot_be_cleared_fails_alone(tmp_path: Path) -> None:
    source = folder(tmp_path, {OTHER.name: OTHER, CLASSKICK[0].name: CLASSKICK[0]})
    workspace = Workspace(tmp_path / "ws")
    first = ingest(source, workspace)
    [key] = [r.student_key for r in first.students if r.source_kind == "other"]
    locked = workspace.student_dir(key) / "locked"
    locked.mkdir()
    (locked / "kept").write_text("")
    locked.chmod(0o500)
    resubmitted = Canvas(str(source / OTHER.name))  # a change, so the folder is cleared
    resubmitted.drawString(72, 720, "resubmitted")
    resubmitted.save()

    try:
        summary = ingest(source, workspace)
    finally:
        locked.chmod(0o700)

    assert [r.source_kind for r in summary.students] == ["classkick"]
    assert [f.file for f in summary.failed] == [OTHER.name]


def test_a_key_is_checked_before_anything_is_deleted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = Workspace(tmp_path / "ws")
    outside = workspace.root / "outside"
    outside.mkdir(parents=True)
    (outside / "kept").write_text("")
    monkeypatch.setattr(ingest_module, "_key", lambda given: "../outside")

    with pytest.raises(ValueError, match="StudentKey|pattern"):
        ingest(folder(tmp_path, {OTHER.name: OTHER}), workspace)

    assert (outside / "kept").exists()


def test_of_two_files_for_one_student_the_newer_is_used(tmp_path: Path) -> None:
    name = truth_of(CLASSKICK[0]).student_name  # from the Classkick sidebar
    renamed = f"Resubmitted - {name}.pdf"  # from the file name
    source = folder(tmp_path, {CLASSKICK[0].name: CLASSKICK[0], renamed: OTHER})
    os.utime(source / CLASSKICK[0].name, (1_000_000, 1_000_000))
    os.utime(source / renamed, (2_000_000, 2_000_000))

    summary = ingest(source, Workspace(tmp_path / "ws"))

    [record] = summary.students
    assert (record.student_name, record.source_kind) == (name, "other")
    [duplicate] = summary.duplicates
    assert (duplicate.used, duplicate.dropped) == (renamed, [CLASSKICK[0].name])


def test_a_roster_fixes_spellings_and_reports_both_kinds_of_mismatch(
    tmp_path: Path,
) -> None:
    name, other = truth_of(CLASSKICK[0]).student_name, truth_of(OTHER).student_name
    spelled = "  ".join(name.upper().split())
    roster = tmp_path / "roster.csv"
    roster.write_text(f"{spelled},period 2\nNobody Here,period 3\n")
    source = folder(tmp_path, {CLASSKICK[0].name: CLASSKICK[0], OTHER.name: OTHER})
    workspace = Workspace(tmp_path / "ws")
    keys = {r.student_key for r in ingest(source, workspace).students}
    pages = sorted(workspace.students_dir.rglob("*.png"))
    written = [page.stat().st_mtime_ns for page in pages]
    later = [workspace.map_json(key) for key in keys]  # what later stages write
    for path in later:
        path.write_text("{}")
    before = sorted(workspace.students_dir.rglob("*"))

    summary = ingest(source, workspace, roster)

    assert sorted(r.student_name for r in summary.students) == sorted([spelled, other])
    assert summary.not_on_roster == [other]
    assert summary.missing_from_roster == ["Nobody Here"]
    # A roster moves no one and re-renders nothing; the record takes its spelling.
    assert {r.student_key for r in summary.students} == keys
    assert [page.stat().st_mtime_ns for page in pages] == written
    assert sorted(workspace.students_dir.rglob("*")) == before  # nothing deleted
    for record in summary.students:
        on_disk = read_model(workspace.ingest_json(record.student_key), IngestRecord)
        assert on_disk.student_name == record.student_name


def test_a_key_depends_on_its_own_name_alone(tmp_path: Path) -> None:
    name = truth_of(OTHER).student_name
    alone = ingest(
        folder(tmp_path / "a", {OTHER.name: OTHER}), Workspace(tmp_path / "a")
    )
    files: dict[str, Path | bytes] = {
        OTHER.name: OTHER,
        "Title - José Núñez.pdf": OTHER,
        "Title - Jose Nunez.pdf": OTHER,  # slugs as the name above does
    }

    crowd = ingest(folder(tmp_path / "b", files), Workspace(tmp_path / "b"))

    keys = {r.student_name: r.student_key for r in crowd.students}
    digest = sha256(name.encode()).hexdigest()[:6]
    assert keys[name] == alone.students[0].student_key
    assert keys[name] == f"{name.replace(' ', '_')}-{digest}"
    assert keys["José Núñez"] != keys["Jose Nunez"]
    assert keys["José Núñez"].startswith("Jose_Nunez-")
    assert keys["Jose Nunez"].startswith("Jose_Nunez-")


def test_an_encrypted_or_unreadable_pdf_is_reported_and_skipped(
    tmp_path: Path,
) -> None:
    locked = tmp_path / "locked.pdf"
    canvas = Canvas(str(locked), encrypt="secret")
    canvas.drawString(72, 720, "locked")
    canvas.save()
    files = {"locked.pdf": locked, "broken.pdf": b"not a pdf", OTHER.name: OTHER}

    summary = ingest(folder(tmp_path, files), Workspace(tmp_path / "ws"))

    assert [r.student_name for r in summary.students] == [truth_of(OTHER).student_name]
    reasons = {f.file: f.reason for f in summary.failed}
    assert set(reasons) == {"locked.pdf", "broken.pdf"}
    assert "password" in reasons["locked.pdf"].lower()


def test_an_unchanged_submission_is_not_rendered_again(tmp_path: Path) -> None:
    source = folder(tmp_path, {OTHER.name: OTHER})
    workspace = Workspace(tmp_path / "ws")
    first = ingest(source, workspace)
    [record] = first.students
    page = workspace.student_page(record.student_key, 1)
    written = page.stat().st_mtime_ns

    assert ingest(source, workspace) == first
    assert page.stat().st_mtime_ns == written


def test_a_changed_submission_leaves_nothing_of_the_old_one(tmp_path: Path) -> None:
    source = folder(tmp_path, {OTHER.name: OTHER})
    workspace = Workspace(tmp_path / "ws")
    [record] = ingest(source, workspace).students
    stale = workspace.map_json(record.student_key)
    stale.write_text("{}")
    resubmitted = Canvas(str(source / OTHER.name))  # same file name, one page
    resubmitted.drawString(72, 720, "resubmitted")
    resubmitted.save()

    [changed] = ingest(source, workspace).students

    assert changed.student_key == record.student_key
    assert len(changed.pages) == 1 < len(record.pages)
    assert not stale.exists()
    assert len(list((stale.parent / "pages").iterdir())) == 1


@pytest.mark.skipif(
    not os.environ.get("AUTOGRADER2_REAL_DATA"),
    reason="AUTOGRADER2_REAL_DATA is not set",
)
def test_real_submissions_ingest() -> None:
    """Each real `student_submissions` folder, ingested into a temporary workspace
    that is deleted afterwards. Prints counts only, never a name."""
    folders = submission_folders()
    assert folders
    for index, submissions in enumerate(folders, 1):
        pdfs = len([p for p in submissions.rglob("*") if p.suffix.lower() == ".pdf"])
        with tempfile.TemporaryDirectory() as tmp:
            summary = ingest(submissions, Workspace(Path(tmp)))
        classkick = [r for r in summary.students if r.source_kind == "classkick"]
        other = [r for r in summary.students if r.source_kind == "other"]
        for record in classkick:
            assert {(p.width_px, p.height_px) for p in record.pages} == {(1004, 2008)}
            numbers = [p.slide_number for p in record.pages]
            assert numbers == list(range(1, len(numbers) + 1))
            assert record.student_name.strip()
        other_pages = [p for r in other for p in r.pages]
        sizes = sorted((p.width_px, p.height_px) for p in other_pages)
        ppis = sorted(round(p.native_ppi or 0) for p in other_pages)
        print(
            f"\nfolder {index}: {pdfs} PDFs, {len(summary.students)} students"
            f" ({len(classkick)} Classkick, {len(other)} other),"
            f" Classkick page counts {sorted({len(r.pages) for r in classkick})},"
            f" other page counts {sorted(len(r.pages) for r in other)},"
            f" other page sizes {sizes}, other native ppi {ppis},"
            f" {len(summary.duplicates)} duplicates, {len(summary.failed)} failed,"
            f" {summary.ignored_files} ignored, {len(summary.unsafe_members)} unsafe"
        )
