"""The published API contract and the promise that keeps it (2026-10-21).

``docs/api/openapi.json`` is what the AIxLearning developers build on. The
promise made to them: the contract only grows. A field, a route or a value is
never removed, renamed or retyped; a request never needs more than before.
``compatibility_problems`` lists what would break that promise, and
``tests/unit/test_api_contract.py`` fails on any. An addition passes, but the
file must be exported again (``python scripts/ops/export_openapi.py``) so it
stays the contract of the code.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

PUBLISHED = Path(__file__).resolve().parents[3] / "docs" / "api" / "openapi.json"

DESCRIPTION = (
    "AIxLearning integration contract. Only the /api routes. Start from "
    "docs/api/QUICKSTART.md; details in docs/api/Chatbot_API_Guide.md. Authentication: "
    "HTTP Basic with the instance's client name and password, plus X-End-User-Id. "
    "The contract only grows: ignore fields, values and stream events you do not know."
)

_METHODS = ("get", "post", "put", "patch", "delete")


def integration_contract(app: Any) -> dict[str, Any]:
    """The contract of the ``/api`` routes, as published."""
    spec = app.openapi()
    return {
        "openapi": spec["openapi"],
        "info": {**spec["info"], "description": DESCRIPTION},
        "paths": {p: v for p, v in spec["paths"].items() if p.startswith("/api")},
        "components": spec.get("components", {}),
    }


def _shape(schema: Any) -> Any:
    """What a client relies on in a schema: its type, not its words or its values."""
    if not isinstance(schema, dict):
        return schema
    if "$ref" in schema:
        return ("ref", schema["$ref"].rsplit("/", 1)[-1])
    for key in ("anyOf", "oneOf", "allOf"):
        if key in schema:
            return (key, tuple(sorted(map(repr, (_shape(s) for s in schema[key])))))
    kind = schema.get("type")
    if kind == "array":
        return ("array", _shape(schema.get("items", {})))
    if kind is None and "const" in schema:
        return ("const",)
    return (kind, schema.get("format"))


def _values(schema: Any) -> set[str]:
    """The allowed values a schema names, anywhere inside it."""
    if not isinstance(schema, dict):
        return set()
    found = {repr(v) for v in schema.get("enum", [])}
    if "const" in schema:
        found.add(repr(schema["const"]))
    for key in ("anyOf", "oneOf", "allOf"):
        for sub in schema.get(key, []):
            found |= _values(sub)
    if "items" in schema:
        found |= _values(schema["items"])
    return found


def _refs(node: Any, into: set[str]) -> set[str]:
    if isinstance(node, dict):
        if "$ref" in node:
            into.add(node["$ref"].rsplit("/", 1)[-1])
        for value in node.values():
            _refs(value, into)
    elif isinstance(node, list):
        for value in node:
            _refs(value, into)
    return into


def _request_schemas(spec: dict[str, Any]) -> set[str]:
    """The component schemas a client sends, followed through their references."""
    schemas = spec.get("components", {}).get("schemas", {})
    todo = set()
    for item in spec["paths"].values():
        for method in _METHODS:
            if method in item:
                _refs(item[method].get("requestBody", {}), todo)
    seen: set[str] = set()
    while todo:
        name = todo.pop()
        if name not in seen and name in schemas:
            seen.add(name)
            todo |= _refs(schemas[name], set()) - seen
    return seen


def compatibility_problems(published: dict[str, Any], current: dict[str, Any]) -> list[str]:
    """What in ``current`` breaks a client built on ``published``. Empty: only growth."""
    problems: list[str] = []

    for path, item in published["paths"].items():
        now = current["paths"].get(path)
        for method in _METHODS:
            if method not in item:
                continue
            where = f"{method.upper()} {path}"
            if now is None or method not in now:
                problems.append(f"{where}: the route is gone")
                continue
            before, after = item[method], now[method]
            old_params = {(p["name"], p["in"]): p for p in before.get("parameters", [])}
            for param in after.get("parameters", []):
                key = (param["name"], param["in"])
                if key not in old_params and param.get("required"):
                    problems.append(f"{where}: new required parameter {param['name']}")
            for key in old_params:
                if key not in {(p["name"], p["in"]) for p in after.get("parameters", [])}:
                    problems.append(f"{where}: parameter {key[0]} is gone")
            if after.get("requestBody", {}).get("required") and not before.get(
                "requestBody", {}
            ).get("required"):
                problems.append(f"{where}: the body is now required")
            for status, response in before.get("responses", {}).items():
                old = response.get("content", {})
                new = after.get("responses", {}).get(status, {}).get("content", {})
                for media, body in old.items():
                    if media not in new:
                        problems.append(f"{where} {status}: no longer answers {media}")
                    elif _shape(body.get("schema")) != _shape(new[media].get("schema")):
                        problems.append(f"{where} {status}: the body's type changed")

    old_schemas = published.get("components", {}).get("schemas", {})
    new_schemas = current.get("components", {}).get("schemas", {})
    sent = _request_schemas(published) | _request_schemas(current)
    for name, schema in old_schemas.items():
        now = new_schemas.get(name)
        if now is None:
            problems.append(f"schema {name}: gone")
            continue
        old_props, new_props = schema.get("properties", {}), now.get("properties", {})
        for prop, prop_schema in old_props.items():
            if prop not in new_props:
                problems.append(f"{name}.{prop}: gone")
                continue
            if _shape(prop_schema) != _shape(new_props[prop]):
                problems.append(f"{name}.{prop}: type changed")
            lost = _values(prop_schema) - _values(new_props[prop])
            if lost and _values(new_props[prop]):
                problems.append(f"{name}.{prop}: values no longer accepted {sorted(lost)}")
        old_required, new_required = set(schema.get("required", [])), set(now.get("required", []))
        if name in sent:
            for prop in sorted(new_required - old_required):
                problems.append(f"{name}.{prop}: a request now needs it")
        for prop in sorted(old_required - new_required):
            if name not in sent:
                problems.append(f"{name}.{prop}: a response may now leave it out")
    return problems
