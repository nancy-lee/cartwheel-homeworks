"""HW2 authentication and span-attribute tests. No Langfuse, Docker, or model key."""

from __future__ import annotations

import pytest
from fastapi import HTTPException

from observability.instrument import _set_permission_denied_attributes
from server import app as server_app


@pytest.fixture(autouse=True)
def clean_sessions():
    server_app._SESSIONS.clear()
    yield
    server_app._SESSIONS.clear()


def test_session_rejects_role_that_differs_from_database(world: dict) -> None:
    # User 1 is a shopper in the database; claiming merchant must be refused.
    with pytest.raises(HTTPException) as exc:
        server_app.create_session(server_app.SessionCreate(user_id=1, role="merchant"))
    assert exc.value.status_code == 403
    assert not server_app._SESSIONS


def test_token_for_one_session_cannot_authorize_another(world: dict) -> None:
    first = server_app.create_session(server_app.SessionCreate(user_id=1, role="shopper"))
    second = server_app.create_session(server_app.SessionCreate(user_id=9002, role="merchant"))

    with pytest.raises(HTTPException) as exc:
        server_app._authorize(second["session_id"], f"Bearer {first['token']}")
    assert exc.value.status_code == 403

    # The same token still works for the session it was issued for.
    ctx = server_app._authorize(first["session_id"], f"Bearer {first['token']}")
    assert (ctx.user_id, ctx.role) == (1, "shopper")


class _FakeSpan:
    def __init__(self) -> None:
        self.attributes: dict = {}

    def set_attribute(self, key: str, value) -> None:
        self.attributes[key] = value


def test_permission_denied_attributes() -> None:
    denied, allowed = _FakeSpan(), _FakeSpan()
    _set_permission_denied_attributes(
        denied, {"ok": False, "error": "permission_denied", "reason": "wrong store"}
    )
    _set_permission_denied_attributes(allowed, {"ok": True})
    assert denied.attributes == {
        "cartwheel.permission_denied": True,
        "cartwheel.permission_denied.reason": "wrong store",
    }
    assert allowed.attributes == {"cartwheel.permission_denied": False}
