from __future__ import annotations

import uuid
from typing import Any

import streamlit as st
from langgraph.types import Command

from helpdesk.auth import authenticate_user
from helpdesk.evaluation import replay_session, run_evaluation, save_session
from helpdesk.graph import build_graph
from helpdesk.privacy import redact_pii
from helpdesk.registry import ToolRegistry, list_mock_tickets, load_agents
from helpdesk.settings import PROJECT_ROOT
from helpdesk.telemetry import (
    invoke_with_trace,
    resume_with_trace,
    trace_snapshot,
)

st.set_page_config(page_title="IT Helpdesk Agent", layout="wide")
st.title("IT Helpdesk Multi-Agent System")
st.caption("Gemini · LangGraph · Chroma · OpenTelemetry")

if "authenticated_user" not in st.session_state:
    st.subheader("Sign in")
    with st.form("login_form"):
        user_id = st.text_input("User ID")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Sign in", type="primary")
    if submitted:
        user = authenticate_user(user_id, password)
        if user is None:
            st.error("Invalid user ID or password.")
        else:
            st.session_state["authenticated_user"] = user
            st.rerun()
    st.info(
        "Demo login only. This local mock database does not authenticate "
        "real organizational accounts."
    )
    st.stop()

authenticated_user = st.session_state["authenticated_user"]
st.sidebar.write(
    f"Signed in: **{authenticated_user['display_name']}** "
    f"({authenticated_user['user_id']})"
)
if st.sidebar.button("Sign out"):
    for key in (
        "authenticated_user",
        "active_result",
        "active_session_id",
        "active_config",
        "active_query",
        "conversation_history",
        "active_assistant_recorded",
    ):
        st.session_state.pop(key, None)
    st.rerun()


@st.cache_resource
def get_graph():
    return build_graph()


def _graphviz_source(compiled_graph: Any) -> str:
    diagram = compiled_graph.get_graph()

    def quote(value: str) -> str:
        return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'

    lines = [
        "digraph helpdesk {",
        '  graph [rankdir=LR, bgcolor="transparent", pad="0.3", nodesep="0.5"];',
        '  node [shape=box, style="rounded,filled", fillcolor="#f2f0ff", color="#7c6fbd"];',
        '  edge [color="#6b7280", arrowsize="0.8"];',
    ]
    for node_id, node in diagram.nodes.items():
        label = {"__start__": "START", "__end__": "END"}.get(
            node_id, node.name.replace("_", " ").title()
        )
        shape = "oval" if node_id in {"__start__", "__end__"} else "box"
        lines.append(
            f"  {quote(node_id)} [label={quote(label)}, shape={shape}];"
        )
    for edge in diagram.edges:
        attributes = ' [style="dashed", label="conditional route"]' if edge.conditional else ""
        lines.append(
            f"  {quote(edge.source)} -> {quote(edge.target)}{attributes};"
        )
    lines.append("}")
    return "\n".join(lines)


graph = get_graph()

with st.sidebar:
    st.subheader("Registries")
    st.json(load_agents(PROJECT_ROOT / "config" / "agents.json"))
    st.json(
        ToolRegistry(PROJECT_ROOT / "config" / "tools.yaml").definitions()
    )

tab_chat, tab_graph, tab_tickets, tab_eval, tab_replay, tab_traces = st.tabs(
    ["Helpdesk", "Agent Graph", "Mock Tickets", "Evaluation", "Replay", "OpenTelemetry"]
)

with tab_chat:
    if st.button("New conversation"):
        for key in (
            "conversation_history",
            "active_result",
            "active_session_id",
            "active_config",
            "active_query",
            "active_assistant_recorded",
        ):
            st.session_state.pop(key, None)
        st.rerun()

    conversation_history = st.session_state.setdefault("conversation_history", [])
    query = st.chat_input("Describe your IT issue or ask a follow-up")
    if query and query.strip():
        previous_messages = list(conversation_history)
        conversation_history.append({"role": "user", "content": query.strip()})
        session_id = uuid.uuid4().hex
        config = {"configurable": {"thread_id": session_id}}
        st.session_state["active_session_id"] = session_id
        st.session_state["active_config"] = config
        st.session_state["active_query"] = redact_pii(query.strip())
        st.session_state["active_assistant_recorded"] = False
        result = invoke_with_trace(
            graph,
            {
                "user_query": st.session_state["active_query"],
                "session_id": session_id,
                "tool_events": [],
                "user_id": authenticated_user["user_id"],
                "conversation_history": previous_messages,
            },
            config,
        )
        st.session_state["active_result"] = result

    result = st.session_state.get("active_result")
    if result:
        if result.get("__interrupt__"):
            request = result["__interrupt__"][0].value
            st.warning(request["message"])
            st.write(f"Proposed action: `{request['tool_name']}`")
            approve_col, deny_col = st.columns(2)
            if approve_col.button("Approve mock ticket"):
                result = resume_with_trace(
                    graph,
                    Command(resume={"approved": True}),
                    st.session_state["active_config"],
                    session_id=st.session_state["active_session_id"],
                    trace_id=result["trace_id"],
                )
                st.session_state["active_result"] = result
                st.session_state["active_assistant_recorded"] = False
            if deny_col.button("Deny"):
                result = resume_with_trace(
                    graph,
                    Command(resume={"approved": False}),
                    st.session_state["active_config"],
                    session_id=st.session_state["active_session_id"],
                    trace_id=result["trace_id"],
                )
                st.session_state["active_result"] = result
                st.session_state["active_assistant_recorded"] = False
        elif result.get("response"):
            if not st.session_state.get("active_assistant_recorded"):
                conversation_history.append(
                    {"role": "assistant", "content": result["response"]}
                )
                st.session_state["active_assistant_recorded"] = True
            st.json(
                {
                    "plan": result.get("plan"),
                    "grounding": result.get("grounding"),
                    "tool_events": result.get("tool_events", []),
                }
            )
            if st.button("Save session capture"):
                saved_path = save_session(
                    result,
                    session_id=st.session_state["active_session_id"],
                    user_query=st.session_state["active_query"],
                    conversation_history=conversation_history,
                )
                st.success(f"Saved replay capture to {saved_path}")

    for message in conversation_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

with tab_graph:
    st.subheader("LangGraph workflow")
    st.caption(
        "Dashed edges are conditional routes. The action node pauses for "
        "approval when a ticket is requested."
    )
    st.graphviz_chart(_graphviz_source(graph), width="stretch")

with tab_tickets:
    st.subheader("Synthetic ticket table")
    st.caption("Local demo tickets only; no external ticketing system is connected.")
    tickets = list_mock_tickets()
    if tickets:
        st.dataframe(tickets, width="stretch", hide_index=True)
    else:
        st.info("No approved mock tickets have been created yet.")

with tab_eval:
    st.write(
        "Runs all 50 frozen examples with leave-one-ticket-out retrieval. "
        "The reference resolution is used only by the scorer."
    )
    if st.button("Run evaluation"):
        with st.spinner("Running Gemini batch evaluation..."):
            evaluation = run_evaluation(graph=graph)
        st.session_state["evaluation"] = evaluation
    if "evaluation" in st.session_state:
        evaluation = st.session_state["evaluation"]
        st.json(evaluation["metrics"])
        st.dataframe(evaluation["cases"], use_container_width=True)

with tab_replay:
    capture_path = st.text_input(
        "Capture file path", placeholder="sessions/<session-id>.json"
    )
    if st.button("Replay capture", disabled=not capture_path.strip()):
        result = replay_session(PROJECT_ROOT / capture_path, graph=graph)
        st.markdown(result["response"])
        st.json(
            {
                "plan": result.get("plan"),
                "grounding": result.get("grounding"),
                "tool_events": result.get("tool_events", []),
            }
        )

with tab_traces:
    st.dataframe(trace_snapshot(), use_container_width=True)
    st.caption(
        "Spans are buffered in-process. Set OTEL_CONSOLE_EXPORTER=true to also "
        "export spans to the console."
    )
