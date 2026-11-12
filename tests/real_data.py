"""Where the real exports are, for the tests that run only with REDPEN_REAL_DATA."""

import os
from pathlib import Path


def submission_folders() -> list[Path]:
    """The `student_submissions` folder of REDPEN_REAL_DATA when that is one
    assignment, or of every <term>/<assignment> under it when it is the GRADING
    folder. Never recurses: the data folder also holds `runs` and `feedback`, which
    are off limits."""
    root = Path(os.environ["REDPEN_REAL_DATA"])
    found = [root / "student_submissions", *root.glob("*/*/student_submissions")]
    return sorted(p for p in found if p.is_dir())
