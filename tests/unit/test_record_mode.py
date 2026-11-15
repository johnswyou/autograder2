from pathlib import Path

import pytest

CONFTEST = Path(__file__).parents[1] / "conftest.py"


def test_record_mode_rewrites_a_recording_only_when_its_test_passes(
    pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytester.makeconftest(CONFTEST.read_text())
    recordings = pytester.mkdir("recordings")
    for name in ("failing", "passing"):
        (recordings / f"{name}.jsonl").write_text("a paid recording\n")
    pytester.makepyfile(
        """
        import asyncio

        def use(replay, name, fail):
            async def main():
                async with replay(name):
                    assert not fail, "failed midway"
            asyncio.run(main())

        def test_failing(replay):
            use(replay, "failing", fail=True)

        def test_passing(replay):
            use(replay, "passing", fail=False)
        """
    )
    monkeypatch.setenv("AUTOGRADER2_RECORD", "1")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")

    result = pytester.runpytest()

    result.assert_outcomes(passed=1, failed=1)
    assert (recordings / "failing.jsonl").read_text() == "a paid recording\n"
    assert (recordings / "passing.jsonl").read_text() == ""  # it made no calls
