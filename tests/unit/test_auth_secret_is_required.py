"""A deployed environment may not sign its tokens with the constant in the repo.

``WEBUI_AUTH_SECRET`` signs session cookies, password-reset links, verification
links and the CSRF tokens. Unset, the code falls back to
``DEV_ONLY_aix_webui_change_me`` — a string published in this repository — and
until now said so in a warning, which is exactly the kind of line a deploy
scrolls past.

The rule pinned here: a refusal where the environment declares itself deployed,
the old warning everywhere else. "Everywhere else" matters as much as the
refusal — a check that fires on a laptop teaches people to set the variable to
whatever silences it.
"""

from __future__ import annotations

import pytest

from aix.webui.auth.manager import (
    _DEV_FALLBACK_SECRET,
    assert_secret_is_real,
    using_dev_fallback_secret,
)

pytestmark = pytest.mark.unit


@pytest.fixture
def on_the_fallback(monkeypatch):
    monkeypatch.setattr("aix.webui.auth.manager._AUTH_SECRET", _DEV_FALLBACK_SECRET)


@pytest.fixture
def with_a_real_secret(monkeypatch):
    monkeypatch.setattr("aix.webui.auth.manager._AUTH_SECRET", "a-long-random-string")


@pytest.mark.parametrize("environment", ["production", "prod", "staging", "pilot", "PRODUCTION"])
def test_a_deployed_environment_refuses_to_start(monkeypatch, on_the_fallback, environment):
    monkeypatch.setenv("ENVIRONMENT", environment)

    with pytest.raises(RuntimeError, match="WEBUI_AUTH_SECRET"):
        assert_secret_is_real()


def test_the_refusal_says_what_to_do_and_what_it_costs(monkeypatch, on_the_fallback):
    monkeypatch.setenv("ENVIRONMENT", "production")

    with pytest.raises(RuntimeError) as refused:
        assert_secret_is_real()

    message = str(refused.value)
    assert "secrets.token_urlsafe" in message, "how to generate one"
    assert "logs every teacher out once" in message, "and what it costs to set it"


@pytest.mark.parametrize("environment", ["development", "dev", "local", "test", ""])
def test_a_laptop_still_starts(monkeypatch, on_the_fallback, environment):
    monkeypatch.setenv("ENVIRONMENT", environment)

    assert_secret_is_real()


def test_an_unset_environment_is_not_treated_as_production(monkeypatch, on_the_fallback):
    """A machine with no ENVIRONMENT is a laptop or a test run, not a deploy."""
    monkeypatch.delenv("ENVIRONMENT", raising=False)

    assert_secret_is_real()


def test_a_real_secret_passes_everywhere(monkeypatch, with_a_real_secret):
    monkeypatch.setenv("ENVIRONMENT", "production")

    assert_secret_is_real()
    assert using_dev_fallback_secret() is False
