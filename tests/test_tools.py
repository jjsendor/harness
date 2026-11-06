import pytest

from coding_harness.tools import MAX_OUTPUT_BYTES, Workspace, build_tools, run_pytest


@pytest.fixture
def workspace(tmp_path):
    return Workspace(tmp_path)


def tool_map(workspace):
    return {tool.name: tool for tool in build_tools(workspace, test_timeout=10)}


@pytest.mark.parametrize("path", ["../outside.py", "/tmp/outside.py", ".git/config", ".env"])
def test_rejects_excluded_or_outside_paths(workspace, path):
    with pytest.raises(ValueError):
        workspace.resolve(path)


def test_resolves_symlinks_before_enforcing_boundary(workspace, tmp_path):
    (tmp_path / "outside").symlink_to(tmp_path.parent, target_is_directory=True)
    (tmp_path / ".env").write_text("secret")
    (tmp_path / "alias").symlink_to(tmp_path / ".env")
    for path in ["outside/file.py", "alias"]:
        with pytest.raises(ValueError):
            workspace.resolve(path)


def test_create_edit_read_and_excluded_listing(workspace):
    tools = tool_map(workspace)
    tools["write_file"].invoke({"path": "src/demo.py", "content": "value = 1\n"})
    with pytest.raises(FileExistsError):
        tools["write_file"].invoke({"path": "src/demo.py", "content": "overwrite"})
    tools["edit_file"].invoke(
        {"path": "src/demo.py", "old_text": "value = 1", "new_text": "value = 2"}
    )
    assert tools["read_file"].invoke({"path": "src/demo.py"}) == "1: value = 2"
    (workspace.root / ".env").write_text("secret")
    (workspace.root / ".git").mkdir()
    assert tools["list_files"].invoke({}) == "src/"


def test_edit_requires_a_unique_match(workspace):
    target = workspace.root / "demo.py"
    target.write_text("same same")
    with pytest.raises(ValueError, match="exactly once"):
        tool_map(workspace)["edit_file"].invoke(
            {"path": "demo.py", "old_text": "same", "new_text": "changed"}
        )
    assert target.read_text() == "same same"


def test_output_is_bounded(workspace):
    (workspace.root / "long.py").write_text("x" * 50_000)
    result = tool_map(workspace)["read_file"].invoke({"path": "long.py"})
    assert result.endswith("[output truncated]")
    assert len(result) < MAX_OUTPUT_BYTES + 100


def test_pytest_runs_and_reports_failure(workspace):
    target = workspace.root / "test_example.py"
    target.write_text("def test_example():\n    assert 1 == 1\n")
    assert "1 passed" in run_pytest(workspace, "test_example.py", timeout=10)
    target.write_text("def test_example():\n    assert 1 == 2\n")
    result = run_pytest(workspace, "test_example.py", timeout=10)
    assert "exit code 1" in result
    assert "1 failed" in result


def test_pytest_times_out(workspace):
    (workspace.root / "test_slow.py").write_text(
        "import time\ndef test_slow():\n    time.sleep(30)\n"
    )
    assert "timed out" in run_pytest(workspace, ".", timeout=0.5)


def test_pytest_does_not_receive_harness_credentials(workspace, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "do-not-propagate")
    (workspace.root / "test_env.py").write_text(
        "import os\ndef test_env():\n    assert 'OPENAI_API_KEY' not in os.environ\n"
    )
    assert "1 passed" in run_pytest(workspace, ".", timeout=10)


def test_env_example_is_allowed(workspace):
    assert workspace.resolve(".env.example") == workspace.root / ".env.example"
