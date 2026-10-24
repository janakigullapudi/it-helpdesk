from __future__ import annotations

import unittest
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch
from uuid import uuid4

from langgraph.types import Command
from pydantic import BaseModel

from helpdesk.graph import build_graph
from helpdesk.schemas import GroundingReview, TriagePlan
from helpdesk.telemetry import invoke_with_trace, resume_with_trace


class _StructuredFake:
    def __init__(
        self, schema: type[BaseModel], plan: TriagePlan, prompts: list[str]
    ) -> None:
        self.schema = schema
        self.plan = plan
        self.prompts = prompts

    def invoke(self, prompt: str) -> BaseModel:
        self.prompts.append(prompt)
        if self.schema is TriagePlan:
            return self.plan
        if self.schema is GroundingReview:
            return GroundingReview(grounded=True)
        raise AssertionError(f"Unexpected structured output schema: {self.schema}")


class _FakeLLM:
    def __init__(
        self,
        plan: TriagePlan,
        response_content: Any = "Try signing in again.",
        response_metadata: dict[str, Any] | None = None,
        prompts: list[str] | None = None,
    ) -> None:
        self.plan = plan
        self.response_content = response_content
        self.response_metadata = response_metadata or {}
        self.prompts = prompts if prompts is not None else []

    def with_structured_output(self, schema: type[BaseModel]) -> _StructuredFake:
        return _StructuredFake(schema, self.plan, self.prompts)

    def invoke(self, prompt: str) -> Any:
        self.prompts.append(prompt)
        return type(
            "FakeMessage",
            (),
            {
                "content": self.response_content,
                "response_metadata": self.response_metadata,
            },
        )()


class GraphTests(unittest.TestCase):
    def _run_graph(
        self,
        plan: TriagePlan,
        response_content: Any = "Try signing in again.",
        response_metadata: dict[str, Any] | None = None,
        conversation_history: list[dict[str, str]] | None = None,
        user_query: str = "Please escalate my login issue.",
    ) -> tuple[Any, str, dict[str, Any]]:
        graph = build_graph()
        session_id = uuid4().hex
        prompts: list[str] = []
        fake_llm = _FakeLLM(
            plan,
            response_content,
            response_metadata,
            prompts,
        )
        config = {"configurable": {"thread_id": session_id}}
        state = {
            "user_query": user_query,
            "session_id": session_id,
            "trace_id": "",
            "tool_events": [],
            "user_id": "E1001",
            "conversation_history": conversation_history or [],
        }
        with (
            patch("helpdesk.graph._llm", return_value=fake_llm),
            patch(
                "helpdesk.graph.search_knowledge",
                return_value=[
                    {
                        "ticket_id": "REF-1",
                        "evidence": "Example resolution: reset credentials.",
                        "distance": 0.1,
                    }
                ],
            ),
        ):
            result = invoke_with_trace(graph, state, config)
        return graph, session_id, result

    def test_access_check_runs_as_read_only_tool(self) -> None:
        plan = TriagePlan(
            intent="access",
            issue_summary="Sign-in issue",
            needs_access_check=True,
            create_ticket=False,
            rationale="The request describes a login problem.",
        )
        _, _, result = self._run_graph(plan)
        self.assertTrue(result["access_result"]["identity_verified"])
        self.assertFalse(result["access_result"]["access_changed"])
        self.assertEqual(result["tool_events"][0]["tool_name"], "check_access")
        self.assertNotIn("user_id", result["tool_events"][0]["request"])
        self.assertEqual(result["response"], "Try signing in again.")
        self.assertTrue(result["trace_id"])

    def test_response_accepts_gemini_text_content_blocks(self) -> None:
        plan = TriagePlan(
            intent="other",
            issue_summary="General issue",
            needs_access_check=False,
            create_ticket=False,
            rationale="The request is not access-related.",
        )
        _, _, result = self._run_graph(
            plan,
            response_content=[{"type": "text", "text": "Try restarting the app."}],
        )
        self.assertEqual(result["response"], "Try restarting the app.")

    def test_follow_up_uses_conversation_for_triage_retrieval_and_response(self) -> None:
        plan = TriagePlan(
            intent="software",
            issue_summary="VPN issue",
            needs_access_check=False,
            create_ticket=False,
            rationale="The user is following up about VPN connectivity.",
        )
        prior_messages = [
            {"role": "user", "content": "My VPN disconnects every few minutes."},
            {"role": "assistant", "content": "What operating system are you using?"},
        ]
        graph = build_graph()
        session_id = uuid4().hex
        prompts: list[str] = []
        fake_llm = _FakeLLM(plan, prompts=prompts)
        with (
            patch("helpdesk.graph._llm", return_value=fake_llm),
            patch(
                "helpdesk.graph.search_knowledge",
                return_value=[],
            ) as search,
        ):
            invoke_with_trace(
                graph,
                {
                    "user_query": "Windows 11.",
                    "session_id": session_id,
                    "trace_id": "",
                    "tool_events": [],
                    "conversation_history": prior_messages,
                },
                {"configurable": {"thread_id": session_id}},
            )
        self.assertTrue(any("My VPN disconnects every few minutes." in p for p in prompts))
        self.assertTrue(any("What operating system are you using?" in p for p in prompts))
        self.assertIn("Windows 11.", search.call_args.args[0])
        self.assertIn("VPN disconnects", search.call_args.args[0])

    def test_empty_gemini_response_includes_finish_reason(self) -> None:
        plan = TriagePlan(
            intent="other",
            issue_summary="General issue",
            needs_access_check=False,
            create_ticket=False,
            rationale="The request is not access-related.",
        )
        with self.assertRaisesRegex(ValueError, "finish_reason='SAFETY'"):
            self._run_graph(
                plan,
                response_content="",
                response_metadata={"finish_reason": "SAFETY"},
            )

    def test_ticket_requires_interrupt_and_replays_without_tools(self) -> None:
        plan = TriagePlan(
            intent="other",
            issue_summary="Escalate login problem",
            needs_access_check=False,
            create_ticket=True,
            rationale="The user explicitly requested escalation.",
        )
        graph, session_id, pending = self._run_graph(plan)
        self.assertTrue(pending["__interrupt__"])
        self.assertEqual(
            pending["__interrupt__"][0].value["tool_name"], "create_mock_ticket"
        )
        self.assertEqual(pending.get("tool_events", []), [])

        config = {"configurable": {"thread_id": session_id}}
        with tempfile.TemporaryDirectory() as directory:
            with (
                patch("helpdesk.graph._llm", return_value=_FakeLLM(plan)),
                patch(
                    "helpdesk.registry.get_settings",
                    return_value=SimpleNamespace(data_dir=Path(directory)),
                ),
            ):
                approved = resume_with_trace(
                    graph,
                    Command(resume={"approved": True}),
                    config,
                    session_id=session_id,
                    trace_id=pending["trace_id"],
                )
        self.assertEqual(approved["ticket_result"]["status"], "created")
        self.assertEqual(
            approved["tool_events"][0]["tool_name"], "create_mock_ticket"
        )
        self.assertEqual(approved["tool_events"][0]["request"]["user_id"], "E1001")

        replay_nodes = {
            item["node"]: item["output"] for item in approved["node_history"]
        }
        replay_session = uuid4().hex
        replay_graph = build_graph()
        with (
            patch("helpdesk.graph._llm", side_effect=AssertionError("Gemini called")),
            patch(
                "helpdesk.graph.search_knowledge",
                side_effect=AssertionError("Chroma called"),
            ),
        ):
            replayed = invoke_with_trace(
                replay_graph,
                {
                    "user_query": "Please escalate my login issue.",
                    "session_id": replay_session,
                    "tool_events": [],
                    "replay_mode": True,
                    "replay_nodes": replay_nodes,
                },
                {"configurable": {"thread_id": replay_session}},
            )
        self.assertEqual(replayed["response"], approved["response"])
        self.assertEqual(replayed["ticket_result"], approved["ticket_result"])


if __name__ == "__main__":
    unittest.main()
