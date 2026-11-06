import pytest
from langchain.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver

from coding_harness.agent import build_agent
from coding_harness.cli import main, run_turn
from coding_harness.config import Settings
from coding_harness.tools import Workspace
from coding_harness.tracing import Tracing


def test_missing_credentials_has_actionable_error(monkeypatch, capsys):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(SystemExit) as error:
        main(["run", "--no-tracing", "Hello"])
    assert error.value.code == 1
    assert "Set OPENAI_API_KEY" in capsys.readouterr().err


def test_tracing_requires_keys():
    settings = Settings(
        _env_file=None,
        openai_api_key="fake",
        tracing=True,
        langfuse_public_key=None,
        langfuse_secret_key=None,
    )
    with pytest.raises(ValueError, match="LANGFUSE_PUBLIC_KEY"):
        settings.validate_credentials()


def test_streamed_cli_turn_returns_final_response(tmp_path, model_factory, capsys):
    settings = Settings(tracing=False)
    agent = build_agent(
        settings,
        Workspace(tmp_path),
        InMemorySaver(),
        model=model_factory(AIMessage(content=[{"type": "text", "text": "Done."}])),
    )
    tracing = Tracing(settings)
    assert run_turn(agent, settings, "cli-test", "Hello", tracing) == "Done."
    assert "langfuse" not in capsys.readouterr().err
