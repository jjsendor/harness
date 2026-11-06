import pytest
from langchain.agents.middleware.model_call_limit import ModelCallLimitExceededError
from langchain.agents.middleware.tool_call_limit import ToolCallLimitExceededError
from langchain.messages import AIMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver

from coding_harness.agent import build_agent, checkpoint_id
from coding_harness.config import Settings
from coding_harness.tools import Workspace


def call(name, args, call_id="call-1"):
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": call_id}])


def test_agent_creates_code_runs_real_tests_and_finishes(tmp_path, model_factory):
    code = "def test_added():\n    assert sum([1, 2]) == 3\n"
    model = model_factory(
        call("write_file", {"path": "test_added.py", "content": code}),
        call("run_tests", {"path": "test_added.py"}, "call-2"),
        AIMessage(content="Added a test; pytest passed."),
    )
    agent = build_agent(Settings(tracing=False), Workspace(tmp_path), InMemorySaver(), model=model)
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "Add and test a Python file"}]},
        {"configurable": {"thread_id": "test"}},
    )
    assert (tmp_path / "test_added.py").read_text() == code
    tool_results = [m.content for m in result["messages"] if isinstance(m, ToolMessage)]
    assert "1 passed" in tool_results[-1]
    assert result["messages"][-1].content == "Added a test; pytest passed."


def test_tool_errors_are_recoverable(tmp_path, model_factory):
    model = model_factory(
        call("read_file", {"path": "../secret"}), AIMessage(content="Path denied.")
    )
    agent = build_agent(Settings(tracing=False), Workspace(tmp_path), InMemorySaver(), model=model)
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "Read an invalid file"}]},
        {"configurable": {"thread_id": "errors"}},
    )
    assert result["messages"][-2].status == "error"
    assert "inside the workspace" in result["messages"][-2].content


def test_model_budget_stops_loop(tmp_path, model_factory):
    model = model_factory(call("list_files", {}), AIMessage(content="unreachable"))
    agent = build_agent(
        Settings(tracing=False, max_model_calls=1),
        Workspace(tmp_path),
        InMemorySaver(),
        model=model,
    )
    with pytest.raises(ModelCallLimitExceededError):
        agent.invoke(
            {"messages": [{"role": "user", "content": "Loop"}]},
            {"configurable": {"thread_id": "limit"}},
        )


def test_tool_budget_blocks_excess_calls(tmp_path, model_factory):
    model = model_factory(
        call("list_files", {}),
        call("write_file", {"path": "blocked.py", "content": "blocked"}, "call-2"),
    )
    agent = build_agent(
        Settings(tracing=False, max_tool_calls=1),
        Workspace(tmp_path),
        InMemorySaver(),
        model=model,
    )
    with pytest.raises(ToolCallLimitExceededError):
        agent.invoke(
            {"messages": [{"role": "user", "content": "Loop"}]},
            {"configurable": {"thread_id": "limit"}},
        )
    assert not (tmp_path / "blocked.py").exists()


def test_sqlite_history_survives_agent_restart_and_sessions_are_isolated(tmp_path, model_factory):
    workspace = Workspace(tmp_path)
    database = str(tmp_path / "state.sqlite")
    config = {"configurable": {"thread_id": "persistent"}}
    with SqliteSaver.from_conn_string(database) as saver:
        agent = build_agent(
            Settings(tracing=False),
            workspace,
            saver,
            model=model_factory(AIMessage(content="First reply")),
        )
        agent.invoke({"messages": [{"role": "user", "content": "First task"}]}, config)
    with SqliteSaver.from_conn_string(database) as saver:
        agent = build_agent(
            Settings(tracing=False),
            workspace,
            saver,
            model=model_factory(AIMessage(content="Second reply")),
        )
        result = agent.invoke({"messages": [{"role": "user", "content": "Second task"}]}, config)
        assert [m.content for m in result["messages"]] == [
            "First task",
            "First reply",
            "Second task",
            "Second reply",
        ]
        assert not agent.get_state({"configurable": {"thread_id": "other"}}).values


def test_checkpoint_names_are_scoped_to_workspace(tmp_path):
    assert checkpoint_id(tmp_path, "main") != checkpoint_id(tmp_path / "other", "main")
