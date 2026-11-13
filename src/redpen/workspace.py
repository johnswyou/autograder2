"""Paths inside an assignment's workspace directory (design section 11).

Attribute names follow the file names, with the dot replaced by an underscore.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Self

from pydantic import BaseModel


def write_model(path: Path, model: BaseModel) -> None:
    """Write `model` to `path` as JSON, replacing any earlier file in one step."""
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    partial.write_text(model.model_dump_json(indent=2) + "\n", encoding="utf-8")
    partial.replace(path)


def read_model[M: BaseModel](path: Path, model: type[M]) -> M:
    return model.model_validate_json(path.read_bytes())


@dataclass(frozen=True)
class Workspace:
    root: Path

    @classmethod
    def for_blank(cls, blank_pdf: Path, workspace: Path | None = None) -> Self:
        """`workspace` if given, else `<blank file stem>.redpen` beside the blank."""
        return cls(workspace or blank_pdf.with_suffix(".redpen"))

    @property
    def config_lock_toml(self) -> Path:
        return self.root / "config.lock.toml"

    # key/

    @property
    def blank_pdf(self) -> Path:
        return self.root / "key" / "blank.pdf"

    def key_page(self, number: int) -> Path:
        """Rendering of page `number` of the blank, counting from 1."""
        return self.root / "key" / "pages" / f"p{number:02d}.png"

    def key_page_json(self, number: int) -> Path:
        """What the parser read on page `number` of the blank."""
        return self.root / "key" / "pages" / f"p{number:02d}.json"

    @property
    def spec_json(self) -> Path:
        return self.root / "key" / "spec.json"

    @property
    def spec_rejected_json(self) -> Path:
        """A spec that failed its checks, kept for correction by hand."""
        return self.root / "key" / "spec.rejected.json"

    @property
    def key_yaml(self) -> Path:
        return self.root / "key" / "key.yaml"

    @property
    def solutions_md(self) -> Path:
        return self.root / "key" / "solutions.md"

    @property
    def rubric_md(self) -> Path:
        return self.root / "key" / "rubric.md"

    @property
    def key_report_json(self) -> Path:
        return self.root / "key" / "key_report.json"

    # students/<student_key>/

    @property
    def ingest_summary_json(self) -> Path:
        """The latest ingest's summary: the students every later stage takes."""
        return self.root / "ingest_summary.json"

    @property
    def students_dir(self) -> Path:
        return self.root / "students"

    def student_dir(self, student: str) -> Path:
        return self.students_dir / student

    def source_pdf(self, student: str) -> Path:
        return self.student_dir(student) / "source.pdf"

    def student_page(self, student: str, number: int) -> Path:
        """Rendering of page `number` of the student's submission, counting from 1."""
        return self.student_dir(student) / "pages" / f"p{number:02d}.png"

    def student_page_json(self, student: str, number: int) -> Path:
        """What the mapper found on page `number` of the student's submission."""
        return self.student_dir(student) / "pages" / f"p{number:02d}.json"

    def ingest_json(self, student: str) -> Path:
        return self.student_dir(student) / "ingest.json"

    def fingerprint_json(self, student: str) -> Path:
        return self.student_dir(student) / "fingerprint.json"

    def map_json(self, student: str) -> Path:
        return self.student_dir(student) / "map.json"

    def crop_png(self, student: str, part: str, index: int) -> Path:
        """Zoomed region `index` of `part`, as sent to a model."""
        return self.student_dir(student) / "crops" / f"{part}-{index}.png"

    def reading_json(self, student: str, problem: str) -> Path:
        return self.student_dir(student) / "readings" / f"{problem}.json"

    def grade_json(self, student: str, problem: str) -> Path:
        return self.student_dir(student) / "grades" / f"{problem}.json"

    # audit/, overrides and cache

    def audit_json(self, part: str) -> Path:
        return self.root / "audit" / f"{part}.json"

    @property
    def overrides_yaml(self) -> Path:
        return self.root / "overrides.yaml"

    @property
    def cache_sqlite(self) -> Path:
        return self.root / "cache.sqlite"

    @property
    def run_log(self) -> Path:
        """The traceback of each failure a run recorded, appended."""
        return self.root / "run.log"

    # out/

    @property
    def scores_by_problem_csv(self) -> Path:
        return self.root / "out" / "scores_by_problem.csv"

    @property
    def scores_by_page_csv(self) -> Path:
        return self.root / "out" / "scores_by_page.csv"

    @property
    def scores_detail_csv(self) -> Path:
        return self.root / "out" / "scores_detail.csv"

    @property
    def feedback_dir(self) -> Path:
        return self.root / "out" / "feedback"

    def feedback_md(self, student: str) -> Path:
        """Named by the student key, never the display name: that comes from PDF
        text or a file name, and is untrusted."""
        return self.feedback_dir / f"{student}.md"

    @property
    def review_md(self) -> Path:
        return self.root / "out" / "review.md"

    @property
    def run_report_md(self) -> Path:
        return self.root / "out" / "run_report.md"

    @property
    def eval_md(self) -> Path:
        return self.root / "out" / "eval.md"

    @property
    def eval_json(self) -> Path:
        return self.root / "out" / "eval.json"
