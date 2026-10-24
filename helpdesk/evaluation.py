from __future__ import annotations

import json
import re
import time
import uuid
from pathlib import Path
from typing import Any

from langgraph.types import Command

from helpdesk.graph import build_graph
from helpdesk.privacy import redact_pii
from helpdesk.settings import get_settings
from helpdesk.telemetry import invoke_with_trace, resume_with_trace, trace_snapshot


def load_golden_dataset() -> list[dict[str, str]]:
    path = get_settings().data_dir / "golden_subset.json"
    records = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(records, list) or not 20 <= len(records) <= 50:
        raise ValueError(f"Expected 20-50 frozen evaluation examples in {path}.")
    return records


def token_f1(prediction: str, reference: str) -> float:
    normalize = lambda text: re.findall(r"[a-z0-9]+", text.lower())
    predicted = normalize(prediction)
    expected = normalize(reference)
    if not predicted or not expected:
        return 0.0
    remaining: dict[str, int] = {}
    for token in expected:
        remaining[token] = remaining.get(token, 0) + 1
    matches = 0
    for token in predicted:
        if remaining.get(token, 0) > 0:
            matches += 1
            remaining[token] -= 1
    precision = matches / len(predicted)
    recall = matches / len(expected)
    return 2 * precision * recall / (precision + recall) if matches else 0.0


def _tool_appropriateness(state: dict[str, Any]) -> float:
    for event in state.get("tool_events", []):
        if event["tool_name"] not in {"check_access", "create_mock_ticket"}:
            return 0.0
        if event["tool_name"] == "create_mock_ticket" and not state.get(
            "plan", {}
        ).get("create_ticket"):
            return 0.0
    return 1.0


def run_evaluation(*, graph: Any | None = None) -> dict[str, Any]:
    runner = graph or build_graph()
    rows: list[dict[str, Any]] = []
    for example in load_golden_dataset():
        session_id = f"eval-{example['ticket_id']}-{uuid.uuid4().hex[:8]}"
        config = {"configurable": {"thread_id": session_id}}
        state: dict[str, Any] = {
            "user_query": redact_pii(example["initial_message"]),
            "session_id": session_id,
            "trace_id": "",
            "tool_events": [],
            "exclude_ticket_id": example["ticket_id"],
        }
        started = time.perf_counter()
        result = invoke_with_trace(runner, state, config)
        while result.get("__interrupt__"):
            result = resume_with_trace(
                runner,
                Command(resume={"approved": False}),
                config,
                session_id=session_id,
                trace_id=result["trace_id"],
            )
        duration_ms = (time.perf_counter() - started) * 1000
        rows.append(
            {
                "ticket_id": example["ticket_id"],
                "correctness_f1": token_f1(
                    result["response"], example["resolution_summary"]
                ),
                "groundedness": float(result["grounding"]["grounded"]),
                "tool_appropriateness": _tool_appropriateness(result),
                "latency_ms": round(duration_ms, 2),
                "response": result["response"],
                "grounding_review": result["grounding"],
            }
        )

    return {
        "dataset_size": len(rows),
        "metrics": {
            "correctness_f1": round(
                sum(row["correctness_f1"] for row in rows) / len(rows), 4
            ),
            "groundedness": round(
                sum(row["groundedness"] for row in rows) / len(rows), 4
            ),
            "tool_appropriateness": round(
                sum(row["tool_appropriateness"] for row in rows) / len(rows), 4
            ),
            "mean_latency_ms": round(
                sum(row["latency_ms"] for row in rows) / len(rows), 2
            ),
        },
        "cases": rows,
    }


def save_session(
    result: dict[str, Any],
    *,
    session_id: str,
    user_query: str,
    conversation_history: list[dict[str, str]] | None = None,
) -> Path:
    session_dir = get_settings().session_dir
    session_dir.mkdir(parents=True, exist_ok=True)
    replay_nodes = {
        entry["node"]: entry["output"] for entry in result.get("node_history", [])
    }
    capture = {
        "session_id": session_id,
        "trace_id": result.get("trace_id"),
        "user_query": redact_pii(user_query),
        "conversation_history": [
            {
                "role": message["role"],
                "content": redact_pii(message["content"]),
            }
            for message in conversation_history or []
        ],
        "response": result.get("response"),
        "plan": result.get("plan"),
        "retrieved": result.get("retrieved", []),
        "tool_events": result.get("tool_events", []),
        "grounding": result.get("grounding"),
        "spans": [
            span
            for span in trace_snapshot()
            if span["attributes"].get("session.id") == session_id
        ],
        "replay_nodes": replay_nodes,
    }
    path = session_dir / f"{session_id}.json"
    path.write_text(json.dumps(capture, indent=2), encoding="utf-8")
    return path


def replay_session(path: Path, *, graph: Any | None = None) -> dict[str, Any]:
    capture = json.loads(path.read_text(encoding="utf-8"))
    required = {"session_id", "user_query", "replay_nodes"}
    if not required.issubset(capture) or not isinstance(capture["replay_nodes"], dict):
        raise ValueError(f"Invalid session capture: {path}")
    runner = graph or build_graph()
    session_id = f"replay-{uuid.uuid4().hex}"
    return invoke_with_trace(
        runner,
        {
            "user_query": capture["user_query"],
            "session_id": session_id,
            "tool_events": [],
            "replay_mode": True,
            "replay_nodes": capture["replay_nodes"],
            "conversation_history": capture.get("conversation_history", []),
        },
        {"configurable": {"thread_id": session_id}},
    )
