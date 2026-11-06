from uuid import uuid4

from langchain.messages import AIMessage
from langfuse import Langfuse
from langgraph.checkpoint.memory import InMemorySaver
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from coding_harness.agent import build_agent
from coding_harness.cli import run_turn
from coding_harness.config import Settings
from coding_harness.tools import Workspace
from coding_harness.tracing import Tracing


def test_langfuse_exports_nested_agent_model_and_tool_spans(tmp_path, model_factory, monkeypatch):
    exporter = InMemorySpanExporter()
    monkeypatch.setattr(
        "coding_harness.tracing.Langfuse",
        lambda **kwargs: Langfuse(**kwargs, span_exporter=exporter),
    )
    settings = Settings(
        tracing=True,
        langfuse_public_key=f"pk-lf-test-{uuid4().hex}",
        langfuse_secret_key="sk-lf-test",
    )
    model = model_factory(
        AIMessage(content="", tool_calls=[{"name": "list_files", "args": {}, "id": "one"}]),
        AIMessage(content="Inspected workspace."),
    )
    agent = build_agent(settings, Workspace(tmp_path), InMemorySaver(), model=model)
    tracing = Tracing(settings)
    try:
        assert run_turn(agent, settings, "trace-test", "Inspect", tracing) == "Inspected workspace."
        tracing.client.flush()
        spans = exporter.get_finished_spans()
        kinds = {span.attributes.get("langfuse.observation.type") for span in spans}
        assert {"generation", "tool"} <= kinds
        assert any(span.name == "list_files" for span in spans)
        assert any(span.attributes.get("session.id") == "trace-test" for span in spans)
        assert len({span.context.trace_id for span in spans}) == 1
        assert any(span.parent is not None for span in spans)
    finally:
        tracing.close()
