# IT Helpdesk Multi-Agent System

A small, reviewable helpdesk demo using LangGraph, Chroma, Google Gemini,
OpenTelemetry, a CLI, and Streamlit.

## Safety and data boundaries

- `data/golden_subset.json` is a frozen 50-ticket sample generated from the
  supplied CSV. It contains only `ticket_id`, `initial_message`, and
  `resolution_summary`. Customer identifiers, contact/CSAT fields, and other
  metadata are excluded.
- At inference time, only the redacted `initial_message` is sent as the user
  request. References are used by evaluation and by the example-based knowledge
  index; evaluation excludes the current ticket from retrieval.
- Email addresses and common phone-number formats are redacted before Gemini
  calls and session capture. Use synthetic input only.
- Access checks are mock, read-only responses. The only write-like action is a
  local mock ticket persisted in `data/mock_tickets.sqlite`; the policy
  interrupts the graph and requires approval.
  No account changes or external tickets are possible.
- Session captures can contain the redacted user query, model response, tool
  inputs/outputs, and retrieved evidence. Keep the local `sessions/` directory
  private; it is git-ignored.

## Setup (Windows PowerShell)

Use Python 3.12 (Python 3.11-3.14 is supported; Python 3.15 is not yet
supported by the current dependency stack):

```powershell
cd .\it-helpdesk-agent
py -3.12 -m venv .venv
# If using uv instead: uv venv --python 3.12 .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

On Windows, Chroma's native extension also requires the Microsoft Visual C++
2015-2022 Redistributable (x64). Install it if Chroma reports
`DLL load failed while importing chromadb_rust_bindings`.

Put your Google AI Studio API key in `.env` as `GEMINI_API_KEY=...`.
The default models are `gemini-2.5-flash` and `models/gemini-embedding-001`;
both can be changed in `.env`.
If Gemini returns no response text, the error includes available finish-reason
or safety metadata. Check that the configured model is enabled for your API
key; if the finish reason indicates a safety block, rephrase the request.

## Demo sign-in

The Streamlit app has a local demo login backed by `config/mock_users.json`.
Passwords are stored as salted PBKDF2 hashes; the app never sends or saves the
password. Use one of these synthetic accounts:

| User ID | Password |
| --- | --- |
| `E1001` | `AlexDemo!2026` |
| `E1002` | `JordanDemo!2026` |
| `E1003` | `SamDemo!2026` |

This mock login only identifies users in this demo. The mock access check can
confirm that a signed-in user exists in the local mock directory, but it cannot
inspect or change real accounts. Do not reuse these example passwords or use
this local login for a public or production deployment.

Approved mock tickets are stored in the SQLite `mock_tickets` table in
`data/mock_tickets.sqlite`. Each row includes its generated ID, signed-in mock
user ID (when available), summary, redacted request, status, and UTC creation
time. Ticket writes still require explicit human approval and do not reach an
external ticketing system. View created rows in the Streamlit **Mock Tickets**
tab.

The Helpdesk tab supports multi-turn chat. Follow-up messages use the recent
conversation as context for triage, knowledge retrieval, and response
generation. Use **New conversation** to clear the visible chat and start over.
Saved session captures include the conversation transcript for review.
The **Agent Graph** tab displays the compiled LangGraph workflow, including
conditional routes.

## Run

```powershell
python -m helpdesk.cli agents
python -m helpdesk.cli ask "I cannot sign in with my 2FA code"
python -m helpdesk.cli ask "Please escalate this issue and open a ticket" --approve
python -m helpdesk.cli eval
python -m helpdesk.cli eval --save-baseline
python -m helpdesk.cli replay .\sessions\<session-id>.json
python -m helpdesk.cli traces
streamlit run app.py
```

The CLI asks for approval interactively when a terminal is available. Without
an interactive terminal, a requested mock ticket is denied unless `--approve`
is explicitly passed. The Streamlit approval buttons resume the paused graph.
LangGraph's in-memory checkpointer is used for this demo; pending approvals do
not survive a process restart.

## Architecture

1. **Triage/planning** classifies the request with a typed Gemini response.
2. **Knowledge search** embeds the query with Gemini and searches the persistent
   Chroma collection. The frozen examples are indexed on first use.
3. **Access-check** runs only for access-related requests and returns a mock
   result without verifying identity or changing permissions.
4. **Policy/action** routes any requested mock ticket through a LangGraph human
   interrupt. The registry validates Pydantic inputs/outputs and rejects
   unapproved ticket calls.
5. **Response/reflection** generates an evidence-grounded answer and checks it;
   if unsupported specifics are found, the answer is rewritten.
6. **Observability** emits OpenTelemetry spans for graph agents, policy
   decisions, and tools. Spans are buffered for the dashboard and optionally
   printed to console via `OTEL_CONSOLE_EXPORTER=true`. CLI runs include their
   spans in output and the session capture.
7. **Replay** reruns graph routing and state transitions from recorded node
   outputs, without calling Gemini, Chroma, or executing tools again.

Tool contracts live in `config/tools.yaml`; the discoverable agent catalogue is
in `config/agents.json`.

## Evaluation

The batch runner evaluates all 50 cases, passing only each `initial_message`
to the graph and withholding its own resolution from retrieval. It reports:

- `correctness_f1`: token-level F1 against `resolution_summary` (a lexical
  baseline, not a semantic judge).
- `groundedness`: the structured Gemini reflection result (a model judgment,
  not a formal proof).
- `tool_appropriateness`: a policy/registry invariant check that no unregistered
  tool or unrequested ticket action was used.
- `mean_latency_ms`.

`eval --save-baseline` saves `data/baseline.json`; a later `eval` reports metric
deltas. Evaluation denies any ticket approval request, preventing batch runs
from making even mock tickets by default.
