"""A whole run of the quiz class (tests/fixtures/masking) from a mock OpenRouter, once
with names masked and once without: `prepare` from the blank, `grade` with the
second read, the second grade and the class audit, `regrade` after a criterion is
edited, and `key verify` (design 14.1).

No request holds a student's name as printed, a file name or a student key, in its
URL, its headers or its body, session id included. Every chat request goes to its
role's pinned model on exactly that model's pinned endpoints, with data collection
denied and zero data retention required. With names masked, every image of a
student's page sent after the map is the unmasked run's image with the names
blanked in flat black and nothing else changed."""

import base64
import functools
import io
import json
import re
from pathlib import Path
from typing import Any

import httpx
import pytest
from PIL import Image, ImageChops
from typer.testing import CliRunner

from redpen.cli import app
from redpen.config import Lock, write_lock
from redpen.grade.ingest import IngestSummary
from redpen.llm.client import Gateway
from redpen.llm.models import ModelInfo, recheck_providers
from redpen.prepare.key import load_key, save_key
from redpen.workspace import Workspace, read_model
from tests.fixture_truth import QUIZ, QUIZ_SUBMISSIONS, truth_of
from tests.grader import grades
from tests.pinned import (
    CHECKER,
    GRADER,
    MAPPER,
    PARSER,
    READER,
    REFEREE,
    SECOND,
    SOLVER,
    ZDR,
)

MODELS = {
    "parser": PARSER,
    "solver": SOLVER,
    "checker": CHECKER,
    "mapper": MAPPER,
    "reader": READER,
    "grader": GRADER,
    "second": SECOND,
    "referee": REFEREE,
}
# The roles that ask for each response model. A request goes to one of their models.
ASKED_BY = {
    "PageParse": ("parser",),
    "Spec": ("parser",),
    "Solution": ("solver", "checker"),
    "Rulings": ("referee",),
    "Rubric": ("solver",),
    "Grades": ("grader",),
    "Decisions": ("grader",),
    "Inventory": ("mapper",),
    "NamedInventory": ("mapper",),
    "Resolution": ("mapper",),
    "Readings": ("reader", "second"),
}
BLOCK = [0.05, 0.05, 0.95, 0.5]  # the one block of work the mock finds on a page
NAME = [0.6, 0.01, 0.9, 0.04]  # and the name it finds there, clear of the block
IMAGE = re.compile(rb"data:image/[a-z]+;base64,[A-Za-z0-9+/=]+")


class OpenRouter:
    """Answers every request of a whole run and keeps them all. The checker solves
    a numeric part differently from the solver, so that the referee is asked; the
    first grade request of a problem earns every criterion and later ones lose the
    first, so that the class audit is asked (the self-test's requests, which carry
    no image, earn every criterion); the reader is unsure of every part and
    the second reader is not."""

    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []
        self.graded: dict[bytes, dict[str, Any]] = {}  # a grade request asked again
        self.first: set[str] = set()  # the problems a grade request was answered for

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if request.url.path.endswith("/endpoints/zdr"):
            return httpx.Response(200, json={"data": ZDR})
        body = json.loads(request.content)
        texts = [
            json.loads(c["text"])
            for c in body["messages"][1]["content"]
            if c["type"] == "text" and c["text"].startswith(("{", "["))
        ]
        answer = getattr(self, kind(body))(body, texts)
        return httpx.Response(
            200,
            json={
                "model": body["model"],
                "choices": [
                    {
                        "message": {"role": "assistant", "content": json.dumps(answer)},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "cost": 0},
            },
        )

    def PageParse(self, body: Any, texts: list[Any]) -> Any:  # noqa: N802
        return {"items": [], "shared_stems": []}

    def Spec(self, body: Any, texts: list[Any]) -> Any:  # noqa: N802
        return json.loads((QUIZ / "spec.json").read_text())

    def Solution(self, body: Any, texts: list[Any]) -> Any:  # noqa: N802
        value = 2.0 if body["model"] == CHECKER.slug else 1.0
        solved: list[dict[str, Any]] = []
        for part in texts[0]["parts"]:
            answer: dict[str, Any] = dict.fromkeys(("value", "unit", "rel_tol"))
            answer |= {"expression": None, "choices": [], "accept_also": []}
            answer |= {"elements": [], "must_include": []}
            if part["options"]:
                answer["choices"] = part["options"][:1]
            else:
                answer["value"] = value
            solved.append(
                {
                    "part_id": part["id"],
                    "solution": "Work.",
                    "answer": answer,
                    "computation": "" if part["options"] else str(value),
                    "uses": [],
                    "alternatives": [],
                }
            )
        return {"parts": solved}

    def Rulings(self, body: Any, texts: list[Any]) -> Any:  # noqa: N802
        parts = [p for t in texts for p in t.get("parts", []) if "A" in p]
        return {
            "rulings": [
                {"part_id": p["id"], "reason": "A.", "ruling": "A"} for p in parts
            ]
        }

    def Rubric(self, body: Any, texts: list[Any]) -> Any:  # noqa: N802
        criteria = [
            {"kind": "method", "requirement": "Sets it up.", "partial": []},
            {"kind": "final_answer", "requirement": "Right.", "partial": []},
        ]
        return {
            "parts": [
                {"part_id": p["id"], "criteria": criteria, "common_errors": []}
                for p in texts[0]["parts"]
            ]
        }

    def Grades(self, body: Any, texts: list[Any]) -> Any:  # noqa: N802
        if (asked := request_text(body)) in self.graded:  # the second grade
            return self.graded[asked]
        problem = ",".join(t["grade"]["part_id"] for t in texts if "grade" in t)
        loses = problem in self.first
        if b"image_url" in asked:  # a student's work, not the self-test's
            self.first.add(problem)

        def lose(reading: dict[str, Any], criterion: dict[str, Any]) -> str | None:
            return "Not set up." if loses and criterion["id"].endswith("c1") else None

        found = grades(body, lose).model_dump(mode="json")
        self.graded[asked] = found
        return found

    def Decisions(self, body: Any, texts: list[Any]) -> Any:  # noqa: N802
        [cases] = [t["cases"] for t in texts if "cases" in t]
        return {
            "decisions": [
                {"case": c["case"], "reason": "r", "awarded": 1} for c in cases
            ]
        }

    def Inventory(self, body: Any, texts: list[Any]) -> Any:  # noqa: N802
        block = {"box": BLOCK, "label": "", "description": "work", "candidates": []}
        return {"rotation": 0, "photo": False, "blocks": [block]}

    def NamedInventory(self, body: Any, texts: list[Any]) -> Any:  # noqa: N802
        return self.Inventory(body, texts) | {"names": [NAME]}

    def Resolution(self, body: Any, texts: list[Any]) -> Any:  # noqa: N802
        parts, blocks = texts[0]["parts"], texts[1]["blocks"]
        mine = {
            p["id"]: [b for b in blocks if b["page"] == p["home_page"]] for p in parts
        }
        entries = [
            {
                "part_id": id,
                "block_id": found[0]["id"],
                "basis": "position",
                "role": "work",
                "confidence": "high",
            }
            for id, found in mine.items()
            if found
        ]
        claimed = {e["block_id"] for e in entries}
        return {
            "entries": entries,
            "unanswered": [id for id, found in mine.items() if not found],
            "set_aside": [
                {"block_id": b["id"], "reason": "other"}
                for b in blocks
                if b["id"] not in claimed
            ],
        }

    def Readings(self, body: Any, texts: list[Any]) -> Any:  # noqa: N802
        unsure = body["model"] != SECOND.slug
        return {
            "readings": [
                {
                    "part_id": part["id"],
                    "status": "answered",
                    "medium": "handwritten",
                    "lines": ["x = 4"],
                    "final_answer": {
                        "choices": part["options"][:1],
                        "value": None,
                        "unit": None,
                        "expression": None,
                    },
                    "elements": [],
                    "uncertain": [1] if unsure else [],
                    "addressed_to_grader": [],
                    "legibility": "clear",
                }
                for part in texts[0]["parts"]
            ]
        }

    @property
    def chats(self) -> list[dict[str, Any]]:
        return [
            json.loads(r.content)
            for r in self.requests
            if r.url.path.endswith("/chat/completions")
        ]


def kind(body: dict[str, Any]) -> str:
    """The response model a chat request asks for."""
    return body["response_format"]["json_schema"]["name"]


def request_text(body: dict[str, Any]) -> bytes:
    """A request's messages, images included: what makes it the same request."""
    return json.dumps(body["messages"]).encode()


def run_class(root: Path, masked: bool) -> tuple[OpenRouter, Workspace]:
    """The whole run in `root`, through the command line, each command seen to
    succeed."""
    router = OpenRouter()
    transport = httpx.MockTransport(router)
    gateway = functools.partial(Gateway, transport=transport, requests_per_second=1e6)
    with pytest.MonkeyPatch.context() as patch:
        patch.chdir(root)
        patch.setenv("OPENROUTER_API_KEY", "test-key")
        for where in ("redpen.cli.Gateway", "redpen.orchestrator.Gateway"):
            patch.setattr(where, gateway)
        patch.setattr(
            "redpen.orchestrator.recheck_providers",
            functools.partial(recheck_providers, transport=transport),
        )
        if masked:
            Path("redpen.toml").write_text("mask_names = true\n")
        (root / "blank.pdf").write_bytes((QUIZ / "blank.pdf").read_bytes())
        (root / "subs").mkdir()
        for pdf in QUIZ_SUBMISSIONS:
            (root / "subs" / pdf.name).write_bytes(pdf.read_bytes())
        workspace = Workspace(Path("ws"))
        write_lock(workspace.config_lock_toml, Lock(profile="balanced", models=MODELS))
        on = ["--workspace", "ws"]
        for args in (["prepare", "blank.pdf", *on], ["grade", "subs", *on]):
            done = CliRunner().invoke(app, args)
            assert done.exit_code == 0, done.output
        key = load_key(workspace.key_yaml)
        [q2] = [p for p in key.parts if p.id == "Q2"]
        edited = q2.criteria[0].model_copy(update={"requirement": "Edited."})
        q2 = q2.model_copy(update={"criteria": [edited, *q2.criteria[1:]]})
        save_key(
            workspace.key_yaml,
            key.model_copy(
                update={"parts": [q2 if p.id == "Q2" else p for p in key.parts]}
            ),
        )
        for args in (["regrade", *on], ["key", "verify", *on]):
            done = CliRunner().invoke(app, args)
            assert done.exit_code == 0, done.output
        assert not workspace.run_log.exists()
    return router, Workspace(root / "ws")


@pytest.fixture(scope="module")
def runs(
    tmp_path_factory: pytest.TempPathFactory,
) -> dict[bool, tuple[OpenRouter, Workspace]]:
    """The whole run, without names masked (False) and with them (True)."""
    return {
        m: run_class(tmp_path_factory.mktemp(f"masked-{m}"), m) for m in (False, True)
    }


def model_of(body: dict[str, Any]) -> ModelInfo:
    """The pinned model of the role a chat request is made for."""
    [model] = {
        MODELS[role].slug: MODELS[role]
        for role in ASKED_BY[kind(body)]
        if MODELS[role].slug == body["model"]
    }.values()
    return model


@pytest.mark.parametrize("masked", [False, True], ids=["off", "on"])
def test_every_request_of_a_whole_run_goes_only_to_its_role_s_endpoints_keeping_nothing(
    runs: dict[bool, tuple[OpenRouter, Workspace]], masked: bool
) -> None:
    router, _ = runs[masked]

    chats = router.chats

    asked = {kind(b) for b in chats}
    named = "NamedInventory" if masked else "Inventory"
    assert asked == (set(ASKED_BY) - {"Inventory", "NamedInventory"}) | {named}
    assert {b["model"] for b in chats} == {m.slug for m in MODELS.values()}
    readers = {b["model"] for b in chats if kind(b) == "Readings"}
    assert readers == {READER.slug, SECOND.slug}  # the second read was made
    for body in chats:
        model = model_of(body)
        assert body["provider"] == {
            "require_parameters": True,
            "data_collection": "deny",
            "zdr": True,
            "only": sorted(model.providers),
        }, kind(body)


def everything_but_images(request: httpx.Request) -> str:
    """A request's URL, headers and body, each image in the body left out (its
    base64 could hold any short string by chance; a name in an image is pixels)."""
    headers = "\n".join(f"{k}: {v}" for k, v in request.headers.items())
    body = IMAGE.sub(b"<image>", request.content).decode()
    return f"{request.url}\n{headers}\n{body}"


@pytest.mark.parametrize("masked", [False, True], ids=["off", "on"])
def test_no_request_of_a_whole_run_holds_a_student_s_name_file_name_or_key(
    runs: dict[bool, tuple[OpenRouter, Workspace]], masked: bool
) -> None:
    router, workspace = runs[masked]
    students = read_model(workspace.ingest_summary_json, IngestSummary).students
    forbidden: set[str] = set()
    for pdf in QUIZ_SUBMISSIONS:
        name = truth_of(pdf).student_name
        forbidden |= {name, *name.split(), pdf.name, pdf.stem}
    for student in students:
        key = student.student_key
        forbidden |= {student.student_name, key, key.rsplit("-", 1)[0]}
    assert len(students) == 3 and {s.student_name for s in students} == {
        truth_of(p).student_name for p in QUIZ_SUBMISSIONS
    }

    for request in router.requests:
        text = everything_but_images(request)
        assert not [f for f in forbidden if f in text], request.url
    sessions = {b.get("session_id") for b in router.chats}
    assert sessions and all(s and s.startswith("ws:") for s in sessions)


def images(body: dict[str, Any]) -> list[Image.Image]:
    return [
        Image.open(io.BytesIO(base64.b64decode(c["image_url"]["url"].split(",")[1])))
        for c in body["messages"][1]["content"]
        if c["type"] == "image_url"
    ]


def texts_of(body: dict[str, Any]) -> str:
    """A request but its images: its model, response model, system text and text
    parts."""
    parts = [c["text"] for c in body["messages"][1]["content"] if c["type"] == "text"]
    return json.dumps([body["model"], kind(body), body["messages"][0], parts])


def changed(on: Image.Image, off: Image.Image) -> tuple[int, int, int, int] | None:
    return ImageChops.difference(on.convert("RGB"), off.convert("RGB")).getbbox()


def area(box: tuple[int, int, int, int] | None) -> int:
    return (box[2] - box[0]) * (box[3] - box[1]) if box else 0


def test_with_names_masked_every_page_image_sent_after_the_map_has_them_blanked(
    runs: dict[bool, tuple[OpenRouter, Workspace]],
) -> None:
    plain = runs[False][0].chats
    blanked: dict[str, int] = {}

    for body in runs[True][0].chats:
        if kind(body) not in ("Readings", "Grades") or not images(body):
            continue
        # The same request without names masked: of the requests whose text is the
        # same (one per student), the one whose images differ least.
        alike = min(
            (b for b in plain if texts_of(b) == texts_of(body)),
            key=lambda b: sum(
                area(changed(x, y))
                for x, y in zip(images(body), images(b), strict=True)
            ),
        )
        for on, off in zip(images(body), images(alike), strict=True):
            box = changed(on, off)
            assert box, "an image of a page with a name went unblanked"
            assert on.convert("RGB").crop(box).getextrema() == ((0, 0),) * 3
            role = "second" if body["model"] == SECOND.slug else kind(body)
            blanked[role] = blanked.get(role, 0) + 1
    assert blanked.keys() == {"Readings", "second", "Grades"}


def test_with_names_masked_the_mapper_is_sent_each_page_whole(
    runs: dict[bool, tuple[OpenRouter, Workspace]],
) -> None:
    plain = [b["messages"][1] for b in runs[False][0].chats if kind(b) == "Inventory"]
    looks = [b for b in runs[True][0].chats if kind(b) == "NamedInventory"]

    assert len(looks) == len(plain) == 6  # three students, two pages each
    for body in looks:
        assert body["messages"][1] in plain  # the same page, outline and size
