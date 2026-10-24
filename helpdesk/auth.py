from __future__ import annotations

import hashlib
import hmac
import json
from functools import lru_cache
from typing import Any

from helpdesk.settings import PROJECT_ROOT

_ITERATIONS = 600_000
_DUMMY_SALT = bytes.fromhex("f7b552b9a739ad12d3fb2e76d202217a")
_DUMMY_HASH = bytes.fromhex(
    "2d7958e138fd08df4c978a6d667697173a1ba3e1b9313f632be2a919a05e6e18"
)


@lru_cache(maxsize=1)
def _mock_users() -> dict[str, dict[str, Any]]:
    path = PROJECT_ROOT / "config" / "mock_users.json"
    users = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(users, list) or not users:
        raise ValueError(f"Mock user database is empty or invalid: {path}")

    result: dict[str, dict[str, Any]] = {}
    for user in users:
        required = {
            "user_id",
            "display_name",
            "department",
            "password_salt",
            "password_hash",
            "active",
        }
        if not isinstance(user, dict) or set(user) != required:
            raise ValueError(f"Unexpected fields in mock user database: {path}")
        if not isinstance(user["user_id"], str) or not user["user_id"].strip():
            raise ValueError(f"Invalid user ID in mock user database: {path}")
        key = user["user_id"].casefold()
        if key in result:
            raise ValueError(f"Duplicate user ID in mock user database: {path}")
        try:
            salt = bytes.fromhex(user["password_salt"])
            password_hash = bytes.fromhex(user["password_hash"])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid password hash in mock user database: {path}") from exc
        if len(salt) < 16 or len(password_hash) != hashlib.sha256().digest_size:
            raise ValueError(f"Invalid password hash in mock user database: {path}")
        result[key] = {**user, "_salt": salt, "_hash": password_hash}
    return result


def _password_hash(password: str, salt: bytes) -> bytes:
    return hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), salt, _ITERATIONS
    )


def authenticate_user(user_id: str, password: str) -> dict[str, str] | None:
    user = _mock_users().get(user_id.strip().casefold())
    salt = user["_salt"] if user else _DUMMY_SALT
    expected = user["_hash"] if user else _DUMMY_HASH
    valid_password = hmac.compare_digest(_password_hash(password, salt), expected)
    if user is None or not user["active"] or not valid_password:
        return None
    return {
        "user_id": user["user_id"],
        "display_name": user["display_name"],
        "department": user["department"],
    }


def find_mock_user(user_id: str | None) -> dict[str, str] | None:
    if not user_id:
        return None
    user = _mock_users().get(user_id.strip().casefold())
    if user is None or not user["active"]:
        return None
    return {
        "user_id": user["user_id"],
        "display_name": user["display_name"],
        "department": user["department"],
    }
