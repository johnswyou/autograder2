"""Ingest of a folder or ZIP of submissions (src/redpen/grade/ingest.py)."""

import os
import shutil
import tempfile
import zipfile
from pathlib import Path

import pypdfium2 as pdfium
import pypdfium2.raw as pdfium_c
import pytest
from PIL import Image, ImageChops
from reportlab.pdfgen.canvas import Canvas

from redpen.grade.ingest import IngestSummary, ingest
from redpen.workspace import Workspace
from tests.fixture_truth import SUBMISSIONS, truth_of

OTHER = next(p for p in SUBMISSIONS if truth_of(p).source_kind == "other")
CLASSKICK = [p for p in SUBMISSIONS if truth_of(p).source_kind == "classkick"]


def folder(root: Path, files: dict[str, Path | bytes]) -> Path:
    """A folder under `root` holding each file, copied or written, under its name."""
    path = root / "submissions"
    path.mkdir()
    for name, content in files.items():
        if isinstance(content, Path):
            shutil.copyfile(content, path / name)
        else:
            (path / name).write_bytes(content)
    return path


@pytest.fixture(scope="module")
def fixtures(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[Workspace, IngestSummary]:
    """The fixture submissions, ingested from a folder."""
    root = tmp_path_factory.mktemp("fixtures")
    workspace = Workspace(root / "ws")
    return workspace, ingest(folder(root, {p.name: p for p in SUBMISSIONS}), workspace)


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
    fixtures: tuple[Workspace, IngestSummary], pdf: Path
) -> None:
    workspace, summary = fixtures
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

    summary = ingest(source, Workspace(tmp_path / "ws"), roster)

    assert sorted(r.student_name for r in summary.students) == sorted([spelled, other])
    assert summary.not_on_roster == [other]
    assert summary.missing_from_roster == ["Nobody Here"]


def test_names_that_would_share_a_key_get_different_keys(tmp_path: Path) -> None:
    files: dict[str, Path | bytes] = {
        "Title - José Núñez.pdf": OTHER,
        "Title - Jose Nunez.pdf": OTHER,
    }

    summary = ingest(folder(tmp_path, files), Workspace(tmp_path / "ws"))

    keys = {r.student_name: r.student_key for r in summary.students}
    assert set(keys) == {"José Núñez", "Jose Nunez"}
    assert keys["Jose Nunez"] == "Jose_Nunez"
    assert keys["José Núñez"].startswith("Jose_Nunez-")


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
    not os.environ.get("REDPEN_REAL_DATA"), reason="REDPEN_REAL_DATA is not set"
)
def test_real_submissions_ingest() -> None:
    """Every `student_submissions` folder under REDPEN_REAL_DATA (the GRADING folder
    or one assignment in it), ingested into a temporary workspace that is deleted
    afterwards. Prints counts only, never a name."""
    folders = sorted(Path(os.environ["REDPEN_REAL_DATA"]).rglob("student_submissions"))
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
