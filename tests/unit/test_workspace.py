from pathlib import Path

from redpen.workspace import Workspace


def test_paths_follow_the_design_layout() -> None:
    ws = Workspace(Path("unit4-forces.redpen"))

    assert ws.config_lock_toml.as_posix() == "unit4-forces.redpen/config.lock.toml"
    assert ws.key_yaml.as_posix() == "unit4-forces.redpen/key/key.yaml"
    assert ws.key_page(1).as_posix() == "unit4-forces.redpen/key/pages/p01.png"
    assert (
        ws.student_page("s1", 12).as_posix()
        == "unit4-forces.redpen/students/s1/pages/p12.png"
    )
    assert (
        ws.crop_png("s1", "Q7.b", 1).as_posix()
        == "unit4-forces.redpen/students/s1/crops/Q7.b-1.png"
    )
    assert (
        ws.grade_json("s1", "Q7").as_posix()
        == "unit4-forces.redpen/students/s1/grades/Q7.json"
    )
    assert ws.audit_json("Q7.b").as_posix() == "unit4-forces.redpen/audit/Q7.b.json"
    assert ws.cache_sqlite.as_posix() == "unit4-forces.redpen/cache.sqlite"
    assert (
        ws.scores_by_page_csv.as_posix() == "unit4-forces.redpen/out/scores_by_page.csv"
    )


def test_default_workspace_is_named_after_the_blank_and_sits_beside_it() -> None:
    ws = Workspace.for_blank(Path("/term/unit4/ch4.2-forces.pdf"))

    assert ws.root == Path("/term/unit4/ch4.2-forces.redpen")


def test_workspace_option_overrides_the_default() -> None:
    ws = Workspace.for_blank(Path("/term/unit4/forces.pdf"), Path("/elsewhere/ws"))

    assert ws.root == Path("/elsewhere/ws")
