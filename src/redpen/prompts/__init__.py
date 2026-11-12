"""Model-facing text. `<stage>/<name>.md` is the system text of one kind of call;
everything that varies goes into the call's content parts."""

import hashlib
from importlib.resources import files


def load(stage: str, name: str) -> tuple[str, str]:
    """The text of `<stage>/<name>.md` and its version, a short hash of its bytes."""
    data = files(__name__).joinpath(stage, f"{name}.md").read_bytes()
    return data.decode("utf-8"), hashlib.sha256(data).hexdigest()[:12]
