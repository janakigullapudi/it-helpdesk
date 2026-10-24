from __future__ import annotations

from typing import Any, NotRequired, TypedDict

from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from helpdesk.knowledge import search_knowledge
from helpdesk.privacy import redact_pii
from helpdesk.registry import ToolRegistry
from helpdesk.schemas import GroundingReview, TriagePlan
from helpdesk.settings import PROJECT_ROOT, get_settings, require_api_key
from helpdesk.telemetry import span


class HelpdeskState(TypedDict):
    user_query: str
    session_id: str
    trace_id: str
    plan: NotRequired[dict[str, Any]]
    retrieved: NotRequired[list[dict[str, Any]]]
    access_result: NotRequired[dict[str, Any]]
    ticket_result: NotRequired[dict[str, Any]]
    response: NotRequired[str]
    grounding: NotRequired[dict[str, Any]]
    approval_pending: NotRequired[bool]
    tool_events: NotRequired[list[dict[str, Any]]]
    node_history: NotRequired[list[dict[str, Any]]]
    replay_nodes: NotRequired[dict[str, dict[str, Any]]]
    replay_mode: NotRequired[bool]
    exclude_ticket_id: NotRequired[str]
    user_id: NotRequired[str]
    conversation_history: NotRequired[list[dict[str, str]]]


def _llm() -> ChatGoogleGenerativeAI:
    api_key = require_api_key()
    return ChatGoogleGenerativeAI(
        model=get_settings().model,
        google_api_key=api_key,
        temperature=0,
        max_retries=2,
        timeout=60,
    )


def _response_text(message: Any, stage: str) -> str:
    content = message.content
    if isinstance(content, str):
        text = content
    elif isinstance(content, list):
        parts = [
            part
            if isinstance(part, str)
            else part.get("text", "")
            for part in content
            if isinstance(part, str)
            or (isinstance(part, dict) and isinstance(part.get("text"), str))
        ]
        text = "\n".join(parts)
    else:
        text = ""

    text = text.strip()
    if text:
        return text

    metadata = getattr(message, "response_metadata", {})
    diagnostic_keys = ("finish_reason", "prompt_feedback", "safety_ratings")
    diagnostics = (
        "; ".join(
            f"{key}={metadata[key]!r}" for key in diagnostic_keys if key in metadata
        )
        if isinstance(metadata, dict)
        else ""
    )
    detail = f" Gemini metadata: {diagnostics}." if diagnostics else ""
    raise ValueError(
        f"Gemini returned no text during {stage}.{detail} Check the finish reason "
        "and safety feedback, and confirm the configured model is available."
    )


def _conversation_text(state: HelpdeskState) -> str:
    messages = state.get("conversation_history", [])[-10:]
    return "\n".join(
        f"{message['role'].capitalize()}: {redact_pii(message['content'])}"
        for message in messages
        if message.get("role") in {"user", "assistant"}
        and isinstance(message.get("content"), str)
        and message["content"].strip()
    )


def _registry() -> ToolRegistry:
    return ToolRegistry(PROJECT_ROOT / "config" / "tools.yaml")


def _record(
    state: HelpdeskState, node: str, output: dict[str, Any]
) -> dict[str, Any]:
    history = list(state.get("node_history", []))
    history.append({"node": node, "output": output})
    return {**output, "node_history": history}


def _replay(state: HelpdeskState, node: str) -> dict[str, Any] | None:
    if not state.get("replay_mode"):
        return None
    saved = state.get("replay_nodes", {}).get(node)
    if saved is None:
        raise ValueError(f"Replay capture does not contain node output: {node}")
    return _record(state, node, saved)


def _tool_event(
    state: HelpdeskState, name: str, request: dict[str, Any], response: dict[str, Any]
) -> list[dict[str, Any]]:
    events = list(state.get("tool_events", []))
    events.append({"tool_name": name, "request": request, "response": response})
    return events


def build_graph(*, checkpointer: Any | None = None) -> Any:
    registry = _registry()

    def triage(state: HelpdeskState) -> dict[str, Any]:
        replayed = _replay(state, "triage")
        if replayed is not None:
            return replayed
        with span(
            "agent.triage_planner",
            agent_name="triage_planner",
            session_id=state["session_id"],
            trace_id=state.get("trace_id"),
        ):
            prompt = (
                "Classify this IT support request and propose a plan. "
                "Only set needs_access_check for login, authentication, or "
                "permission issues. Set create_ticket true only if the user "
                "explicitly asks for escalation or a support ticket. Never "
                "claim that an external action has happened. Use the prior "
                "conversation to understand short follow-up messages, but base "
                "the plan on the user's current request.\n\n"
                f"Prior conversation:\n{_conversation_text(state) or '(none)'}\n\n"
                f"Request: {redact_pii(state['user_query'])}"
            )
            plan = _llm().with_structured_output(TriagePlan).invoke(prompt)
            if not isinstance(plan, TriagePlan):
                plan = TriagePlan.model_validate(plan)
            return _record(state, "triage", {"plan": plan.model_dump()})

    def retrieve(state: HelpdeskState) -> dict[str, Any]:
        replayed = _replay(state, "retrieve")
        if replayed is not None:
            return replayed
        with span(
            "agent.knowledge_search",
            agent_name="knowledge_search",
            session_id=state["session_id"],
            trace_id=state.get("trace_id"),
        ):
            prior_user_messages = [
                message["content"]
                for message in state.get("conversation_history", [])[-10:]
                if message.get("role") == "user"
                and isinstance(message.get("content"), str)
            ]
            retrieval_query = "\n".join(
                [*prior_user_messages, state["user_query"]]
            )
            hits = search_knowledge(
                redact_pii(retrieval_query),
                exclude_ticket_id=state.get("exclude_ticket_id"),
                session_id=state["session_id"],
                trace_id=state.get("trace_id"),
            )
            return _record(state, "retrieve", {"retrieved": hits})

    def access_check(state: HelpdeskState) -> dict[str, Any]:
        replayed = _replay(state, "access_check")
        if replayed is not None:
            return replayed
        with span(
            "agent.access_check",
            agent_name="access_check",
            session_id=state["session_id"],
            trace_id=state.get("trace_id"),
        ):
            request = {"query": redact_pii(state["user_query"])}
            result = registry.execute(
                "check_access",
                {**request, "user_id": state.get("user_id")},
                session_id=state["session_id"],
                trace_id=state.get("trace_id"),
            )
            events = _tool_event(state, "check_access", request, result)
            return _record(
                state,
                "access_check",
                {"access_result": result, "tool_events": events},
            )

    def action(state: HelpdeskState) -> dict[str, Any]:
        replayed = _replay(state, "action")
        if replayed is not None:
            return replayed
        with span(
            "agent.action_ticket",
            agent_name="action_ticket",
            session_id=state["session_id"],
            trace_id=state.get("trace_id"),
        ):
            plan = state["plan"]
            if not plan["create_ticket"]:
                return _record(
                    state,
                    "action",
                    {"ticket_result": {"status": "not_requested"}, "approval_pending": False},
                )

            request = {
                "query": redact_pii(state["user_query"]),
                "summary": plan["issue_summary"],
                "user_id": state.get("user_id"),
            }
            approval_required = registry.requires_approval("create_mock_ticket")
            if approval_required:
                with span(
                    "policy.approval_decision",
                    tool_name="create_mock_ticket",
                    decision="human_approval_required",
                    session_id=state["session_id"],
                    trace_id=state.get("trace_id"),
                ):
                    approval = interrupt(
                        {
                            "kind": "tool_approval",
                            "tool_name": "create_mock_ticket",
                            "summary": plan["issue_summary"],
                            "message": (
                                "Approve creation of a local mock ticket? "
                                "No external ticketing system is connected."
                            ),
                        }
                    )
                if not isinstance(approval, dict) or approval.get("approved") is not True:
                    return _record(
                        state,
                        "action",
                        {
                            "ticket_result": {"status": "denied"},
                            "approval_pending": False,
                        },
                    )

            result = registry.execute(
                "create_mock_ticket",
                request,
                approved=approval_required,
                session_id=state["session_id"],
                trace_id=state.get("trace_id"),
            )
            events = _tool_event(state, "create_mock_ticket", request, result)
            return _record(
                state,
                "action",
                {
                    "ticket_result": result,
                    "approval_pending": False,
                    "tool_events": events,
                },
            )

    def respond(state: HelpdeskState) -> dict[str, Any]:
        replayed = _replay(state, "respond")
        if replayed is not None:
            return replayed
        with span(
            "agent.response_generation",
            agent_name="response_reflection",
            session_id=state["session_id"],
            trace_id=state.get("trace_id"),
        ):
            prompt = (
                "You are a careful IT helpdesk assistant. Answer the user with "
                "clear, practical next steps. Use only the provided evidence and "
                "mock access result. Do not claim an account change or ticket "
                "creation unless the supplied result confirms it. If evidence is "
                "insufficient, say so and suggest a safe next step. Keep the "
                "answer concise. Treat the prior conversation as context, answer "
                "the current request, and do not repeat earlier answers unless "
                "needed.\n\n"
                f"Prior conversation:\n{_conversation_text(state) or '(none)'}\n\n"
                f"User request: {redact_pii(state['user_query'])}\n"
                f"Retrieved evidence: {state.get('retrieved', [])}\n"
                f"Access result: {state.get('access_result', 'not needed')}\n"
                f"Ticket result: {state.get('ticket_result', 'not requested')}"
            )
            response = _response_text(_llm().invoke(prompt), "helpdesk response")
            return _record(state, "respond", {"response": response})

    def reflect(state: HelpdeskState) -> dict[str, Any]:
        replayed = _replay(state, "reflect")
        if replayed is not None:
            return replayed
        with span(
            "agent.groundedness_reflection",
            agent_name="response_reflection",
            session_id=state["session_id"],
            trace_id=state.get("trace_id"),
        ):
            evidence = {
                "retrieved": state.get("retrieved", []),
                "access_result": state.get("access_result"),
                "ticket_result": state.get("ticket_result"),
            }
            review = _llm().with_structured_output(GroundingReview).invoke(
                "Check whether the answer's factual claims are supported by the "
                "provided evidence and whether it falsely claims an action. "
                "Treat generic advice as acceptable; flag unsupported specifics.\n"
                f"Evidence: {evidence}\nAnswer: {state['response']}"
            )
            if not isinstance(review, GroundingReview):
                review = GroundingReview.model_validate(review)
            response = state["response"]
            if not review.grounded:
                revised = _llm().invoke(
                    "Rewrite the answer to remove every unsupported claim. "
                    "Use only the evidence and explicitly state uncertainty where "
                    "needed. Do not claim an action unless the tool result confirms it.\n"
                    f"Evidence: {evidence}\nOriginal answer: {response}\n"
                    f"Unsupported claims: {review.unsupported_claims}"
                )
                response = _response_text(revised, "grounded revision")
            result = {
                "response": response,
                "grounding": review.model_dump(),
                "approval_pending": False,
            }
            return _record(state, "reflect", result)

    builder = StateGraph(HelpdeskState)
    builder.add_node("triage", triage)
    builder.add_node("retrieve", retrieve)
    builder.add_node("access_check", access_check)
    builder.add_node("action", action)
    builder.add_node("respond", respond)
    builder.add_node("reflect", reflect)
    builder.add_edge(START, "triage")
    builder.add_edge("triage", "retrieve")
    builder.add_conditional_edges(
        "retrieve",
        lambda state: "access_check" if state["plan"]["needs_access_check"] else "action",
        {"access_check": "access_check", "action": "action"},
    )
    builder.add_edge("access_check", "action")
    builder.add_edge("action", "respond")
    builder.add_edge("respond", "reflect")
    builder.add_edge("reflect", END)
    return builder.compile(checkpointer=checkpointer or MemorySaver())
