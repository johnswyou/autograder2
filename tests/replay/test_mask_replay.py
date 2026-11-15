"""The quiz class (tests/fixtures/masking) mapped with names masked and read from
that map, with recorded calls to the real mapper and reader models. The reads are
chained: each is made from the map the mapper gave, its names blanked.

The inputs are the quiz's own: `blank.pdf`, `spec.json` and a fresh ingest of its
three submissions. The recording, `tests/recordings/mask_quiz.jsonl`, was made by
running this test's calls live into a persistent scratch cache and keeping exactly
the calls made (task 5.3 report). Under `AUTOGRADER2_RECORD=1` the `replay` fixture
would call the models again (paid) and rewrite it once the test passes."""

import asyncio
import shutil
from collections.abc import Callable

import pytest

from autograder2.config import Config
from autograder2.grade.ingest import IngestSummary, ingest
from autograder2.grade.map import map_student
from autograder2.grade.read import read_student
from autograder2.llm.client import Gateway
from autograder2.prepare.parse import render_blank
from autograder2.schemas.reading import ReadingRecord
from autograder2.schemas.spec import Spec
from autograder2.schemas.submission import MapRecord
from autograder2.workspace import Workspace, read_model
from tests.fixture_truth import QUIZ, QUIZ_SUBMISSIONS, truth_of
from tests.pinned import MAPPER, READER
from tests.replay.test_map_replay import coverage

SPEC = read_model(QUIZ / "spec.json", Spec)


@pytest.fixture(scope="module")
def quiz_workspace(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[Workspace, IngestSummary]:
    """A workspace holding the quiz's blank, its page images and `spec.json`, and
    its three submissions ingested."""
    root = tmp_path_factory.mktemp("quiz")
    source = root / "submissions"
    source.mkdir()
    for pdf in QUIZ_SUBMISSIONS:
        shutil.copyfile(pdf, source / pdf.name)
    workspace = Workspace(root / "ws")
    workspace.blank_pdf.parent.mkdir(parents=True)
    shutil.copyfile(QUIZ / "blank.pdf", workspace.blank_pdf)
    shutil.copyfile(QUIZ / "spec.json", workspace.spec_json)
    render_blank(workspace)
    return workspace, ingest(source, workspace)


def test_the_quiz_maps_with_each_name_found_and_reads_as_its_truth_with_it_blanked(
    quiz_workspace: tuple[Workspace, IngestSummary],
    replay: Callable[[str], Gateway],
) -> None:
    workspace, summary = quiz_workspace
    budget = Config().image_budget
    keys = {s.student_name: s.student_key for s in summary.students}

    async def main() -> dict[str, tuple[MapRecord, dict[str, ReadingRecord]]]:
        found: dict[str, tuple[MapRecord, dict[str, ReadingRecord]]] = {}
        async with replay("mask_quiz") as gateway:
            for key in sorted(keys.values()):
                mapped = await map_student(
                    workspace, key, SPEC, gateway, MAPPER, budget, True
                )
                read = await read_student(workspace, key, SPEC, gateway, READER, budget)
                found[key] = (mapped, read)
        return found

    found = asyncio.run(main())

    for pdf in QUIZ_SUBMISSIONS:
        truth = truth_of(pdf)
        mapped, read = found[keys[truth.student_name]]
        # Each name is found on its page and blanked whole; no other name is found.
        names = [(p.number, b) for p in mapped.pages for b in p.names]
        assert all(p.names_on_responses == [] for p in mapped.pages)
        assert len(names) == len(truth.map.set_aside)
        for want in truth.map.set_aside:
            assert max(coverage(b, want.box) for n, b in names if n == want.page) == 1
        # Read from the blanked pages, every part is read as the truth says.
        for want in truth.readings:
            got = read[want.part_id].reading
            assert (got.status, got.final_answer.choices) == (
                want.status,
                want.final_answer.choices,
            ), want.part_id
            assert got.final_answer.value == want.final_answer.value, want.part_id
            assert got.legibility == "clear" and got.uncertain == [], want.part_id
