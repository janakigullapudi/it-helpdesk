from __future__ import annotations

import json
import sqlite3
import uuid
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import yaml
from pydantic import BaseModel

from helpdesk.auth import find_mock_user
from helpdesk.schemas import (
    CheckAccessInput,
    CheckAccessOutput,
    CreateTicketInput,
    CreateTicketOutput,
)
from helpdesk.settings import get_settings
from helpdesk.telemetry import span

_INPUT_MODELS: dict[str, type[BaseModel]] = {
    "CheckAccessInput": CheckAccessInput,
    "CreateTicketInput": CreateTicketInput,
}
_OUTPUT_MODELS: dict[str, type[BaseModel]] = {
    "CheckAccessOutput": CheckAccessOutput,
    "CreateTicketOutput": CreateTicketOutput,
}


def _check_access(payload: BaseModel) -> BaseModel:
    request = CheckAccessInput.model_validate(payload)
    user = find_mock_user(request.user_id)
    if user is None:
        return CheckAccessOutput(
            identity_verified=False,
            access_changed=False,
            message=(
                "No authenticated user was found in the mock directory. "
                "No access was changed."
            ),
        )
    return CheckAccessOutput(
        identity_verified=True,
        access_changed=False,
        message="The signed-in user was verified in the mock directory. No access was changed.",
    )


def _create_mock_ticket(payload: BaseModel) -> BaseModel:
    request = CreateTicketInput.model_validate(payload)
    database_path = get_settings().data_dir / "mock_tickets.sqlite"
    database_path.parent.mkdir(parents=True, exist_ok=True)
    ticket_id = f"MOCK-{uuid.uuid4().hex[:8].upper()}"
    created_at = datetime.now(timezone.utc).isoformat()
    with closing(sqlite3.connect(database_path)) as connection:
        with connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS mock_tickets (
                    ticket_id TEXT PRIMARY KEY,
                    user_id TEXT,
                    summary TEXT NOT NULL,
                    query TEXT NOT NULL,
                    status TEXT NOT NULL CHECK (status IN ('created')),
                    created_at TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                INSERT INTO mock_tickets
                    (ticket_id, user_id, summary, query, status, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    ticket_id,
                    request.user_id,
                    request.summary,
                    request.query,
                    "created",
                    created_at,
                ),
            )
    return CreateTicketOutput(
        ticket_id=ticket_id,
        status="created",
        message=f"Mock ticket created: {request.summary}",
    )


def list_mock_tickets() -> list[dict[str, Any]]:
    database_path = get_settings().data_dir / "mock_tickets.sqlite"
    if not database_path.exists():
        return []
    with closing(sqlite3.connect(database_path)) as connection:
        connection.row_factory = sqlite3.Row
        table = connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
            ("mock_tickets",),
        ).fetchone()
        if table is None:
            return []
        rows = connection.execute(
            """
            SELECT ticket_id, user_id, summary, query, status, created_at
            FROM mock_tickets
            ORDER BY created_at DESC, ticket_id DESC
            """
        ).fetchall()
    return [dict(row) for row in rows]


class ToolRegistry:
    def __init__(self, config_path: Path) -> None:
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        if not isinstance(config, dict) or not isinstance(config.get("tools"), list):
            raise ValueError(f"Invalid tool registry: {config_path}")
        self.version = str(config.get("version", ""))
        self._definitions = {item["name"]: item for item in config["tools"]}
        self._implementations: dict[str, Callable[[BaseModel], BaseModel]] = {
            "check_access": _check_access,
            "create_mock_ticket": _create_mock_ticket,
        }
        for name, definition in self._definitions.items():
            if (
                name not in self._implementations
                or definition.get("input_schema") not in _INPUT_MODELS
                or definition.get("output_schema") not in _OUTPUT_MODELS
                or not definition.get("version")
            ):
                raise ValueError(f"Invalid or unimplemented registered tool: {name}")

    def definitions(self) -> list[dict[str, Any]]:
        return list(self._definitions.values())

    def requires_approval(self, name: str) -> bool:
        if name not in self._definitions:
            raise KeyError(f"Tool is not registered: {name}")
        return bool(self._definitions[name]["approval_required"])

    def execute(
        self,
        name: str,
        payload: dict[str, Any],
        *,
        approved: bool = False,
        session_id: str | None = None,
        trace_id: str | None = None,
    ) -> dict[str, Any]:
        if name not in self._definitions:
            raise KeyError(f"Tool is not registered: {name}")
        definition = self._definitions[name]
        if definition["approval_required"] and not approved:
            raise PermissionError(f"Tool requires human approval: {name}")

        input_model = _INPUT_MODELS[definition["input_schema"]]
        output_model = _OUTPUT_MODELS[definition["output_schema"]]
        with span(
            f"tool.{name}",
            tool_name=name,
            tool_version=str(definition["version"]),
            approval_required=bool(definition["approval_required"]),
            approved=approved,
            session_id=session_id,
            trace_id=trace_id,
        ):
            validated_input = input_model.model_validate(payload)
            result = self._implementations[name](validated_input)
            return output_model.model_validate(result).model_dump()


def load_agents(config_path: Path) -> list[dict[str, Any]]:
    data = json.loads(config_path.read_text(encoding="utf-8"))
    agents = data.get("agents")
    if not isinstance(agents, list):
        raise ValueError(f"Invalid agent registry: {config_path}")
    return agents
