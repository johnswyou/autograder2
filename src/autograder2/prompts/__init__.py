"""Model-facing text. `<stage>/<name>.md` is the system text of one kind of call;
everything that varies goes into the call's content parts."""

import hashlib
from importlib.resources import files


def load(stage: str, name: str) -> tuple[str, str]:
    """The text of `<stage>/<name>.md` and its version, a short hash of its bytes."""
    data = files(__name__).joinpath(stage, f"{name}.md").read_bytes()
    return data.decode("utf-8"), hashlib.sha256(data).hexdigest()[:12]


def versions() -> dict[str, str]:
    """The version of every prompt file, by `<stage>/<name>`, in name order."""
    found = {
        f"{stage.name}/{file.name.removesuffix('.md')}": hashlib.sha256(
            file.read_bytes()
        ).hexdigest()[:12]
        for stage in files(__name__).iterdir()
        if stage.is_dir()
        for file in stage.iterdir()
        if file.name.endswith(".md")
    }
    return dict(sorted(found.items()))
