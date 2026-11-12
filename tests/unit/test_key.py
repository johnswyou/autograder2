import json
import re
import shutil
from collections.abc import Callable
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from redpen.cli import app
from redpen.prepare.key import (
    assemble,
    key_version,
    load_key,
    part_hash,
    rubric_md,
    save_key,
    solutions_md,
)
from redpen.prepare.rubric import PartRubric, ProposedCriterion
from redpen.prepare.solve import Alternative, PartSolution
from redpen.schemas.key import Answer, Criterion, Key, PartialCredit
from redpen.schemas.spec import Part, Spec
from redpen.workspace import Workspace
from tests.fixture_truth import FIXTURES

KEY = Key.model_validate_json((FIXTURES / "key.json").read_text())


def spec_part(id: str, response_type: str = "numeric", **fields: Any) -> Part:
    return Part.model_validate(
        {
            "id": id,
            "label": id[1:],
            "prompt": f"Prompt of {id}.",
            "response_type": response_type,
            "home_page": 1,
            "band": [0.1, 0.2],
            "figures": [],
            "depends_on": [],
            "options": [],
        }
        | fields
    )


def solved(
    id: str, answer: dict[str, Any], uses: list[str] | None = None, *also: float
) -> PartSolution:
    return PartSolution(
        part_id=id,
        solution=f"Work of {id}.",
        answer=Answer.model_validate(answer),
        uses=uses or [],
        alternatives=[
            Alternative(assumption="another constant", value=v) for v in also
        ],
    )


def proposed(kind: str, *levels: str) -> ProposedCriterion:
    return ProposedCriterion.model_validate(
        {
            "kind": kind,
            "requirement": f"A {kind} requirement.",
            "partial": [{"points": p, "when": f"Partly, {p}."} for p in levels],
        }
    )


def test_the_key_is_assembled_from_the_stage_outputs(tmp_path: Path) -> None:
    spec = Spec(
        parts=[
            spec_part("Q1", "multiple_choice", options=["A", "B", "C"]),
            spec_part("Q2", "multiple_select", options=["A", "B", "C"]),
            spec_part("Q3.a"),
            spec_part("Q3.b"),
            spec_part("Q4", depends_on=["Q3.a"]),
        ]
    )
    solutions = {
        "Q1": solved("Q1", {"choices": ["B"]}),
        "Q2": solved("Q2", {"choices": ["A", "C"]}),
        "Q3.a": solved("Q3.a", {"value": 2.0, "unit": "m"}, None, 2.1),
        # Itself, a later part and an unknown id are dropped.
        "Q3.b": solved("Q3.b", {"value": 4.0}, ["Q4", "Q3.b", "Q9", "Q3.a"]),
        "Q4": solved("Q4", {"value": 1.0}, ["Q1"]),
    }
    rubrics = {
        "Q3.a": PartRubric(
            part_id="Q3.a",
            criteria=[proposed("method", "0.6", "0.1", "1"), proposed("final_answer")],
            common_errors=["Halves it."],
        ),
        "Q3.b": PartRubric(
            part_id="Q3.b",
            criteria=[proposed("method", "0.75"), proposed("final_answer")],
            common_errors=[],
        ),
        "Q4": PartRubric(
            part_id="Q4", criteria=[proposed("final_answer")], common_errors=[]
        ),
    }

    key = assemble(spec, solutions, rubrics)
    q1, q2, q3a, q3b, q4 = key.parts

    assert q1.criteria == [
        Criterion(
            id="Q1.c1",
            points=Decimal(1),
            kind="final_answer",
            requirement="Chooses B.",
            partial=[],
            carry_forward=False,
        )
    ]
    assert q2.criteria[0].requirement == "Chooses A and C and no other option."
    assert [c.id for c in q3a.criteria] == ["Q3.a.c1", "Q3.a.c2"]
    assert all(c.points == 1 for p in key.parts for c in p.criteria)
    # 0.6 rounds down to 0.5; 0.1 rounds to zero and 1 is not below 1: both dropped.
    assert q3a.criteria[0].partial == [
        PartialCredit(points=Decimal("0.5"), when="Partly, 0.6.")
    ]
    assert q3b.criteria[0].partial[0].points == Decimal("0.75")
    assert q3a.answer.accept_also == [2.1]
    assert [p.depends_on for p in key.parts] == [[], [], [], ["Q3.a"], ["Q1", "Q3.a"]]
    assert [c.carry_forward for c in q3a.criteria] == [False, False]
    assert [c.carry_forward for c in q3b.criteria] == [False, True]
    assert [c.carry_forward for c in q4.criteria] == [True]
    assert q3a.common_errors == ["Halves it."]
    assert {p.state for p in key.parts} == {"unverified"}
    # Every criterion's points are one Decimal object, never written as an anchor.
    save_key(tmp_path / "key.yaml", key)
    points = re.findall(r"points: (\S+)", (tmp_path / "key.yaml").read_text())
    assert points == ["1", "1", "1", "0.5", "1", "1", "0.75", "1", "1"]


def test_the_key_file_round_trips_in_the_layout_of_the_design(tmp_path: Path) -> None:
    path = tmp_path / "key.yaml"

    save_key(path, KEY)
    text = path.read_text()

    assert load_key(path) == KEY
    assert text.startswith("policies:\n  carry_forward: true\n")
    assert "parts:\n  - id: Q1\n    label: '1'\n" in text
    assert re.search(r"\n    solution: \|-\n      f = mu m g", text)
    assert "points: 2\n" in text and "points: 0.5\n" in text
    # An answer shows only the fields its response type uses.
    assert "    answer:\n      choices:\n        - C\n    solution:" in text
    assert text.count("rel_tol") == 1  # the policy


def test_a_hand_edit_survives_a_load_and_a_save(tmp_path: Path) -> None:
    path = tmp_path / "key.yaml"
    save_key(path, KEY)
    policies, parts = path.read_text().split("parts:\n")
    edited = "# Checked by hand.\nparts:\n" + parts + policies
    edited, raised = re.subn(
        r"(id: Q5\.a\.c1\n\s+points:) 2", r"\1 3  # one more", edited
    )
    assert raised == 1
    path.write_text(edited)

    key = load_key(path)
    save_key(path, key)
    text = path.read_text()

    assert key.parts[6].criteria[0].points == 3
    assert text.startswith("# Checked by hand.\nparts:\n")
    assert "points: 3  # one more\n" in text
    assert text.index("parts:") < text.index("policies:")
    assert load_key(path) == key


@pytest.mark.parametrize("written", ["0.1", "0.10000000000000000001"])
def test_a_point_value_loads_as_the_decimal_written(
    tmp_path: Path, written: str
) -> None:
    path = tmp_path / "key.yaml"
    save_key(path, KEY)
    path.write_text(
        path.read_text().replace("- points: 0.5\n", f"- points: {written}\n")
    )

    level = load_key(path).parts[3].criteria[1].partial[0].points

    assert level == Decimal(written) and str(level) == written


def hashes(key: Key) -> dict[str, str]:
    return {part.id: part_hash(part, key.policies) for part in key.parts}


def test_changing_a_part_changes_its_hash_and_the_version_only() -> None:
    data = KEY.model_dump()
    data["parts"][4]["criteria"][1]["requirement"] = "States a = 2.5 m/s^2."
    changed = Key.model_validate(data)

    before, after = hashes(KEY), hashes(changed)

    assert [id for id in before if before[id] != after[id]] == ["Q4.b"]
    assert key_version(changed) != key_version(KEY)
    stricter = KEY.model_copy(
        update={"policies": KEY.policies.model_copy(update={"rel_tol": 0.01})}
    )
    assert all(hashes(stricter)[id] != before[id] for id in before)


def test_reordering_keys_or_reformatting_the_yaml_changes_no_hash(
    tmp_path: Path,
) -> None:
    path, json_path = tmp_path / "key.yaml", tmp_path / "key.json"
    # JSON is YAML: flow style, keys sorted, points as strings, every answer field.
    data = KEY.model_dump(mode="json")
    json_path.write_text(json.dumps(data, sort_keys=True, indent=1))
    reformatted = load_key(json_path)
    save_key(path, KEY)
    text = path.read_text()
    path.write_text(text.replace("points: 1\n", "points: 1.00\n"))
    renumbered = load_key(path)

    assert "points: 1.00\n" in path.read_text() and text != path.read_text()

    assert hashes(reformatted) == hashes(renumbered) == hashes(KEY)
    assert key_version(reformatted) == key_version(renumbered) == key_version(KEY)


def test_the_markdown_shows_every_part_criterion_page_maximum_and_the_total() -> None:
    solutions, rubric = solutions_md(KEY), rubric_md(KEY)

    for part in KEY.parts:
        assert f"## {part.id}: {part.label}, page {part.home_page}\n" in solutions
        assert part.solution in solutions
        assert f"## {part.id}: {part.label}, page {part.home_page}," in rubric
        for c in part.criteria:
            points = f"{c.points} point{'' if c.points == 1 else 's'}"
            assert f"- {c.id} ({points}, " in rubric
    assert "- Value: 2.5 m/s^2\n" in solutions and "- Choice: C\n" in solutions
    assert "Total: 16 points\n" in rubric
    assert "| 1 | 3 |\n| 2 | 5 |\n| 3 | 4 |\n| 4 | 4 |\n" in rubric
    assert "  - 0.5 points when: Only one of the two horizontal forces" in rubric


runner = CliRunner()


def workspace_with(tmp_path: Path, key: Key) -> Workspace:
    workspace = Workspace(tmp_path / "ws")
    workspace.spec_json.parent.mkdir(parents=True)
    shutil.copy(FIXTURES / "spec.json", workspace.spec_json)
    save_key(workspace.key_yaml, key)
    return workspace


def edited(change: Callable[[dict[str, Any]], object]) -> Key:
    """KEY with `change` applied to its parts, by id."""
    data = KEY.model_dump(mode="json")
    change({part["id"]: part for part in data["parts"]})
    return Key.model_validate(data)


@pytest.mark.parametrize(
    ("key", "problem"),
    [
        (
            edited(lambda p: p["Q4.b"]["criteria"][1].update(id="Q4.a.c3")),
            "criterion Q4.a.c3 is not an id of part Q4.b",
        ),
        (
            edited(lambda p: p["Q4.b"]["criteria"][1].update(id="Q4.b.c1")),
            "criterion Q4.b.c1 appears 2 times",
        ),
        (
            edited(lambda p: p["Q5.c"].update(id="Q5.b", criteria=[])),
            "part Q5.b appears 2 times",
        ),
        (
            edited(lambda p: p["Q5.b"].update(depends_on=["Q5.c"])),
            "Q5.b depends on Q5.c, which is not an earlier part",
        ),
        (
            edited(lambda p: p["Q4.b"]["criteria"][1].update(points="0.3")),
            "Q4.b.c2 is worth 0.3, not a multiple of the point step 0.25",
        ),
        (
            edited(
                lambda p: p["Q4.a"]["criteria"][1]["partial"][0].update(points="0.6")
            ),
            "Q4.a.c2 has a partial level of 0.6, not a multiple of the point step 0.25",
        ),
        (
            edited(lambda p: p["Q4.a"]["criteria"][1]["partial"][0].update(points="1")),
            "Q4.a.c2 has a partial level of 1, not below its 1 point",
        ),
        (
            KEY.model_copy(update={"parts": KEY.parts[:-1]}),
            "Q5.c is in the spec but not in the key",
        ),
        (
            edited(
                lambda p: p["Q5.c"].update(
                    id="Q5.d",
                    criteria=[
                        c | {"id": f"Q5.d.c{n}"}
                        for n, c in enumerate(p["Q5.c"]["criteria"], 1)
                    ],
                )
            ),
            "Q5.d is in the key but not in the spec",
        ),
    ],
    ids=[
        "criterion of another part",
        "duplicate criterion",
        "duplicate part",
        "later dependency",
        "points off the step",
        "partial level off the step",
        "partial level not below",
        "spec part missing",
        "key part not in the spec",
    ],
)
def test_key_check_reports_each_problem_with_exit_code_1(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, key: Key, problem: str
) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    workspace = workspace_with(tmp_path, key)

    result = runner.invoke(app, ["key", "check", "--workspace", str(workspace.root)])

    assert result.exit_code == 1
    assert problem in result.output.splitlines()


def test_key_check_reports_a_schema_error(tmp_path: Path) -> None:
    workspace = workspace_with(tmp_path, KEY)
    text = workspace.key_yaml.read_text()
    workspace.key_yaml.write_text(text.replace("home_page: 4", "home_page: 0", 1))

    result = runner.invoke(app, ["key", "check", "--workspace", str(workspace.root)])

    assert result.exit_code == 1
    assert result.output.startswith(f"{workspace.key_yaml}: parts.7.home_page: ")


def test_key_check_passes_a_clean_key_and_render_writes_the_markdown(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    workspace = workspace_with(tmp_path, KEY)
    root = str(workspace.root)

    checked = runner.invoke(app, ["key", "check", "--workspace", root])
    rendered = runner.invoke(app, ["key", "render", "--workspace", root])

    assert checked.exit_code == 0, checked.output
    assert "Total points: 16\n  page 1: 3\n  page 2: 5\n" in checked.output
    assert rendered.exit_code == 0, rendered.output
    assert workspace.solutions_md.read_text() == solutions_md(KEY)
    assert workspace.rubric_md.read_text() == rubric_md(KEY)
