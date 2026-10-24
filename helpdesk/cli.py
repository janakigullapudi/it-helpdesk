from __future__ import annotations

import argparse
import json
import sys
import uuid
from pathlib import Path
from typing import Any

from langgraph.types import Command

from helpdesk.evaluation import (
    load_golden_dataset,
    replay_session,
    run_evaluation,
    save_session,
)
from helpdesk.graph import build_graph
from helpdesk.privacy import redact_pii
from helpdesk.settings import PROJECT_ROOT, get_settings
from helpdesk.telemetry import invoke_with_trace, resume_with_trace, trace_snapshot


def _result_json(state: dict[str, Any]) -> str:
    return json.dumps(
        {
            "response": state.get("response"),
            "approval_pending": bool(state.get("__interrupt__")),
            "plan": state.get("plan"),
            "grounding": state.get("grounding"),
            "tool_events": state.get("tool_events", []),
            "spans": [
                span
                for span in trace_snapshot()
                if span["attributes"].get("session.id") == state.get("session_id")
            ],
        },
        indent=2,
        ensure_ascii=False,
    )


def _ask(query: str, auto_approve: bool) -> int:
    graph = build_graph()
    session_id = uuid.uuid4().hex
    config = {"configurable": {"thread_id": session_id}}
    safe_query = redact_pii(query)
    result = invoke_with_trace(
        graph,
        {
            "user_query": safe_query,
            "session_id": session_id,
            "tool_events": [],
        },
        config,
    )
    if result.get("__interrupt__"):
        approval = auto_approve
        if not auto_approve and sys.stdin.isatty():
            interrupt_value = result["__interrupt__"][0].value
            answer = input(
                f"{interrupt_value['message']} [y/N] "
            ).strip().lower()
            approval = answer in {"y", "yes"}
        result = resume_with_trace(
            graph,
            Command(resume={"approved": approval}),
            config,
            session_id=session_id,
            trace_id=result["trace_id"],
        )

    print(_result_json(result))
    capture_path = save_session(result, session_id=session_id, user_query=safe_query)
    print(f"Session saved to {capture_path}")
    return 0


def _evaluate(save_baseline: bool) -> int:
    result = run_evaluation()
    baseline_path = PROJECT_ROOT / "data" / "baseline.json"
    baseline_path.parent.mkdir(parents=True, exist_ok=True)
    print(json.dumps({"dataset_size": result["dataset_size"], "metrics": result["metrics"]}, indent=2))
    if save_baseline:
        baseline_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"Baseline saved to {baseline_path}")
    elif baseline_path.exists():
        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
        print("Regression comparison (current minus baseline):")
        print(
            json.dumps(
                {
                    metric: round(value - baseline["metrics"][metric], 4)
                    for metric, value in result["metrics"].items()
                    if metric in baseline.get("metrics", {})
                    and metric != "mean_latency_ms"
                },
                indent=2,
            )
        )
    else:
        print("No baseline found. Use `python -m helpdesk.cli eval --save-baseline`.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="IT Helpdesk multi-agent demo")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ask_parser = subparsers.add_parser("ask", help="Answer one helpdesk question")
    ask_parser.add_argument("query")
    ask_parser.add_argument(
        "--approve",
        action="store_true",
        help="Approve a requested local mock ticket (never a real ticket).",
    )

    eval_parser = subparsers.add_parser("eval", help="Evaluate the frozen 50-ticket set")
    eval_parser.add_argument("--save-baseline", action="store_true")

    replay_parser = subparsers.add_parser("replay", help="Replay a captured session")
    replay_parser.add_argument("capture", type=Path)

    subparsers.add_parser("agents", help="List registered agents and tools")
    subparsers.add_parser("traces", help="Show buffered OpenTelemetry spans")
    args = parser.parse_args()

    if args.command == "ask":
        return _ask(args.query, args.approve)
    if args.command == "eval":
        return _evaluate(args.save_baseline)
    if args.command == "replay":
        result = replay_session(args.capture)
        print(_result_json(result))
        return 0
    if args.command == "agents":
        from helpdesk.registry import ToolRegistry, load_agents
        print(
            json.dumps(
                {
                    "agents": load_agents(PROJECT_ROOT / "config" / "agents.json"),
                    "tools": ToolRegistry(
                        PROJECT_ROOT / "config" / "tools.yaml"
                    ).definitions(),
                    "dataset_examples": len(load_golden_dataset()),
                },
                indent=2,
            )
        )
        return 0
    if args.command == "traces":
        spans = trace_snapshot()
        if not spans:
            session_dir = get_settings().session_dir
            for capture_path in sorted(session_dir.glob("*.json")):
                capture = json.loads(capture_path.read_text(encoding="utf-8"))
                spans.extend(capture.get("spans", []))
        print(json.dumps(spans, indent=2, default=str))
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
