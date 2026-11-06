"""Small coding toolset. Path checks are guardrails, not an OS sandbox."""

import os
import signal
import subprocess
import sys
import tempfile
from pathlib import Path

from langchain.tools import tool
from langchain_core.tools import BaseTool

MAX_FILE_BYTES = 1_000_000
MAX_OUTPUT_BYTES = 24_000
EXCLUDED = {".git", ".venv", "venv", ".harness", "node_modules", "__pycache__"}
EXCLUDED |= {".pytest_cache", ".ruff_cache", ".ssh", ".aws"}


class Workspace:
    def __init__(self, root: Path):
        self.root = root.expanduser().resolve(strict=True)
        if not self.root.is_dir():
            raise ValueError("Workspace must be a directory.")

    @staticmethod
    def _check_parts(path: Path) -> None:
        for part in path.parts:
            if part in EXCLUDED or (
                part.startswith(".env") and part not in {".env.example", ".env.template"}
            ):
                raise ValueError("Path is excluded from agent file access.")
            if part.endswith((".pem", ".key")):
                raise ValueError("Key files are excluded from agent file access.")

    def resolve(self, path: str) -> Path:
        relative = Path(path)
        if relative.is_absolute():
            raise ValueError("Use a path relative to the workspace.")
        self._check_parts(relative)
        resolved = (self.root / relative).resolve()
        if not resolved.is_relative_to(self.root):
            raise ValueError("Path must stay inside the workspace.")
        self._check_parts(resolved.relative_to(self.root))
        return resolved

    def read(self, path: str) -> str:
        target = self.resolve(path)
        with target.open("rb") as handle:
            content = handle.read(MAX_FILE_BYTES + 1)
        if len(content) > MAX_FILE_BYTES:
            raise ValueError("File exceeds the 1 MB tool limit.")
        return content.decode("utf-8")


def run_pytest(
    workspace: Workspace, path: str, timeout: float, python: str = sys.executable
) -> str:
    target = workspace.resolve(path)
    if not target.exists():
        raise ValueError("Test target does not exist.")
    relative = str(target.relative_to(workspace.root))
    # Avoid passing the harness's provider/tracing credentials to the test process.
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("OPENAI_", "LANGFUSE_", "HARNESS_"))
    }
    with tempfile.TemporaryFile() as output:
        process = subprocess.Popen(
            [python, "-m", "pytest", "-q", "--", relative],
            cwd=workspace.root,
            env=env,
            stdout=output,
            stderr=subprocess.STDOUT,
            start_new_session=os.name == "posix",
        )
        timed_out = False
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            if os.name == "posix":
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            else:
                process.kill()
            process.wait()
        output.seek(0)
        data = output.read(MAX_OUTPUT_BYTES + 1)
    text = data[:MAX_OUTPUT_BYTES].decode("utf-8", errors="replace")
    if len(data) > MAX_OUTPUT_BYTES:
        text += "\n[output truncated]"
    status = "timed out" if timed_out else f"exit code {process.returncode}"
    return f"pytest {status}\n{text}"


def build_tools(
    workspace: Workspace, test_timeout: float, test_python: str = sys.executable
) -> list[BaseTool]:
    @tool
    def list_files(directory: str = ".") -> str:
        """List up to 200 entries in a workspace directory. Directories end in /."""
        target = workspace.resolve(directory)
        entries = []
        for entry in sorted(target.iterdir(), key=lambda p: p.name):
            relative = str(entry.relative_to(workspace.root))
            try:
                workspace.resolve(relative)
            except ValueError:
                continue
            entries.append(relative + ("/" if entry.is_dir() else ""))
            if len(entries) == 200:
                entries.append("[entry limit reached]")
                break
        return "\n".join(entries) or "[empty directory]"

    @tool
    def read_file(path: str, start_line: int = 1, end_line: int = 200) -> str:
        """Read UTF-8 source, with line numbers. Request at most 200 lines at a time."""
        if start_line < 1 or end_line < start_line or end_line - start_line >= 200:
            raise ValueError("Choose a valid range of at most 200 lines, starting at 1.")
        lines = workspace.read(path).splitlines()
        result = "\n".join(
            f"{index + 1}: {lines[index]}"
            for index in range(start_line - 1, min(end_line, len(lines)))
        )
        return result[:MAX_OUTPUT_BYTES] + (
            "\n[output truncated]" if len(result) > MAX_OUTPUT_BYTES else ""
        )

    @tool
    def write_file(path: str, content: str) -> str:
        """Create a new UTF-8 file. For existing files, use edit_file instead."""
        if len(content.encode("utf-8")) > MAX_FILE_BYTES:
            raise ValueError("File exceeds the 1 MB tool limit.")
        target = workspace.resolve(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("x", encoding="utf-8") as handle:
            handle.write(content)
        return f"Created {path}"

    @tool
    def edit_file(path: str, old_text: str, new_text: str) -> str:
        """Replace exactly one occurrence of old_text in an existing UTF-8 file."""
        original = workspace.read(path)
        if not old_text or original.count(old_text) != 1:
            raise ValueError("old_text must be nonempty and match exactly once.")
        updated = original.replace(old_text, new_text, 1)
        if len(updated.encode("utf-8")) > MAX_FILE_BYTES:
            raise ValueError("Updated file exceeds the 1 MB tool limit.")
        workspace.resolve(path).write_text(updated, encoding="utf-8")
        return f"Updated {path}"

    @tool
    def run_tests(path: str = ".") -> str:
        """Run pytest on a workspace file or directory, with a time and output limit."""
        return run_pytest(workspace, path, test_timeout, test_python)

    return [list_files, read_file, write_file, edit_file, run_tests]
