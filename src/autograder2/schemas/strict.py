"""Lint a JSON Schema against the limits of a strict-mode response format."""

from typing import Any


def strict_problems(schema: dict[str, Any]) -> list[str]:
    """Return what stops `schema` being sent as a strict response format.

    Reports an object property missing from `required`, an object without
    `additionalProperties: false`, and an `anyOf` or `oneOf` with more than one
    member that is an object or a `$ref` to one. `X | None` passes.
    """
    defs: dict[str, Any] = schema.get("$defs", {})
    problems: list[str] = []

    def is_object(node: dict[str, Any]) -> bool:
        if "$ref" in node:
            node = defs.get(node["$ref"].removeprefix("#/$defs/"), {})
        return node.get("type") == "object"

    def walk(node: dict[str, Any], path: str) -> None:
        if node.get("type") == "object":
            required = node.get("required", [])
            for name in node.get("properties", {}):
                if name not in required:
                    problems.append(f"{path}: property {name!r} is not in required")
            if node.get("additionalProperties") is not False:
                problems.append(f"{path}: additionalProperties is not false")
        for keyword in ("anyOf", "oneOf"):
            if sum(is_object(member) for member in node.get(keyword, [])) > 1:
                problems.append(f"{path}: {keyword} has more than one object member")

        for keyword in ("properties", "$defs"):
            for name, child in node.get(keyword, {}).items():
                walk(child, f"{path}/{keyword}/{name}")
        for keyword in ("anyOf", "oneOf", "allOf", "prefixItems"):
            for index, child in enumerate(node.get(keyword, [])):
                walk(child, f"{path}/{keyword}/{index}")
        for keyword in ("items", "additionalProperties"):
            if isinstance(node.get(keyword), dict):
                walk(node[keyword], f"{path}/{keyword}")

    walk(schema, "#")
    return problems
