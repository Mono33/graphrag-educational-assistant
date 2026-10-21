"""The contract only grows (2026-10-21).

The AIxLearning developers were promised that ``docs/api/openapi.json`` would
never lose a field, a route or a value, nor change a type: new things are only
added. This fails when the code breaks that promise, and when an addition was
made without exporting the file again.
"""

from __future__ import annotations

import copy
import json

import pytest

from aix.api.contract import PUBLISHED, compatibility_problems, integration_contract

pytestmark = pytest.mark.unit


@pytest.fixture(scope="module")
def contracts():
    from aix.api.main import app

    published = json.loads(PUBLISHED.read_text(encoding="utf-8"))
    return published, integration_contract(app)


def test_the_code_keeps_the_published_contract(contracts):
    published, current = contracts

    problems = compatibility_problems(published, current)

    assert not problems, "The code breaks the published API contract:\n" + "\n".join(problems)


def test_the_published_contract_is_the_code_s(contracts):
    published, current = contracts

    assert published == current, (
        "The API grew but docs/api/openapi.json was not exported again: "
        "python scripts/ops/export_openapi.py"
    )


# --- The check itself, on small changes to the published contract


def _changed(contracts, change):
    published, _ = contracts
    current = copy.deepcopy(published)
    change(current)
    return compatibility_problems(published, current)


def _schema(spec, name):
    return spec["components"]["schemas"][name]


def test_an_added_field_or_route_is_growth(contracts):
    def grow(spec):
        _schema(spec, "TurnResult")["properties"]["steps"] = {"type": "array", "items": {}}
        spec["paths"]["/api/v1/jobs"] = {"get": {"responses": {}}}

    assert _changed(contracts, grow) == []


def test_a_removed_field_breaks_it(contracts):
    def remove(spec):
        del _schema(spec, "TurnResult")["properties"]["markdown"]

    assert _changed(contracts, remove) == ["TurnResult.markdown: gone"]


def test_a_changed_type_breaks_it(contracts):
    def retype(spec):
        _schema(spec, "TurnResult")["properties"]["session_id"] = {"type": "integer"}

    assert _changed(contracts, retype) == ["TurnResult.session_id: type changed"]


def test_a_request_that_needs_more_breaks_it(contracts):
    def require(spec):
        _schema(spec, "TurnRequest").setdefault("required", []).append("mode")

    assert _changed(contracts, require) == ["TurnRequest.mode: a request now needs it"]


def test_a_value_no_longer_accepted_breaks_it(contracts):
    def narrow(spec):
        _schema(spec, "TurnRequest")["properties"]["mode"]["enum"] = ["auto", "lesson"]

    assert _changed(contracts, narrow) == [
        "TurnRequest.mode: values no longer accepted [\"'answer'\"]"
    ]


def test_a_removed_route_breaks_it(contracts):
    def remove(spec):
        del spec["paths"]["/api/v1/catalog"]

    assert _changed(contracts, remove) == ["GET /api/v1/catalog: the route is gone"]
