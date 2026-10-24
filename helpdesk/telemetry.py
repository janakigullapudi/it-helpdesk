from __future__ import annotations

import os
import threading
from collections import deque
from contextlib import contextmanager
from typing import Any, Iterator, Sequence

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import (
    ConsoleSpanExporter,
    SimpleSpanProcessor,
    SpanExporter,
    SpanExportResult,
)


class _SpanBuffer(SpanExporter):
    def __init__(self, limit: int = 1000) -> None:
        self._spans: deque[dict[str, object]] = deque(maxlen=limit)
        self._lock = threading.Lock()

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        with self._lock:
            for span in spans:
                self._spans.append(
                    {
                        "name": span.name,
                        "trace_id": format(span.context.trace_id, "032x"),
                        "span_id": format(span.context.span_id, "016x"),
                        "parent_span_id": (
                            format(span.parent.span_id, "016x") if span.parent else None
                        ),
                        "start_time": span.start_time,
                        "duration_ms": (
                            (span.end_time - span.start_time) / 1_000_000
                            if span.end_time is not None and span.start_time is not None
                            else None
                        ),
                        "attributes": dict(span.attributes or {}),
                        "status": span.status.status_code.name,
                    }
                )
        return SpanExportResult.SUCCESS

    def shutdown(self) -> None:
        return None

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return True

    def snapshot(self) -> list[dict[str, object]]:
        with self._lock:
            return list(reversed(self._spans))


_buffer = _SpanBuffer()
_provider = TracerProvider(resource=Resource.create({"service.name": "it-helpdesk"}))
_provider.add_span_processor(SimpleSpanProcessor(_buffer))
if os.getenv("OTEL_CONSOLE_EXPORTER", "false").lower() == "true":
    _provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
trace.set_tracer_provider(_provider)
_tracer = trace.get_tracer("it-helpdesk")


@contextmanager
def span(name: str, **attributes: object) -> Iterator[None]:
    with _tracer.start_as_current_span(name) as active_span:
        for key, value in attributes.items():
            if value is not None:
                active_span.set_attribute(key, value)
        yield


def trace_snapshot() -> list[dict[str, object]]:
    return _buffer.snapshot()


def invoke_with_trace(
    graph: Any, state: dict[str, Any], config: dict[str, Any]
) -> dict[str, Any]:
    session_id = state["session_id"]
    with _tracer.start_as_current_span(
        "helpdesk.run", attributes={"session.id": session_id}
    ) as root:
        trace_id = format(root.get_span_context().trace_id, "032x")
        state["trace_id"] = trace_id
        root.set_attribute("app.trace_id", trace_id)
        return graph.invoke(state, config=config)


def resume_with_trace(
    graph: Any,
    command: Any,
    config: dict[str, Any],
    *,
    session_id: str,
    trace_id: str,
) -> dict[str, Any]:
    with _tracer.start_as_current_span(
        "helpdesk.resume",
        attributes={"session.id": session_id, "app.trace_id": trace_id},
    ):
        return graph.invoke(command, config=config)
