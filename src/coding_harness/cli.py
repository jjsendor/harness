import argparse
import os
import sys
from pathlib import Path
from uuid import uuid4

from langchain.messages import AIMessage, ToolMessage

from coding_harness.agent import checkpoint_id, open_agent
from coding_harness.config import Settings
from coding_harness.tools import Workspace
from coding_harness.tracing import Tracing


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description="Local LangChain + Langfuse Python coding agent")
    commands = root.add_subparsers(dest="command", required=True)
    for name, help_text in [("run", "Complete one task"), ("chat", "Start an interactive session")]:
        command = commands.add_parser(name, help=help_text)
        command.add_argument("--workspace", type=Path, default=Path.cwd())
        command.add_argument("--session", help="Reuse this label to continue a saved conversation")
        command.add_argument("--no-tracing", action="store_true", help="Disable Langfuse")
        command.add_argument("--model", help="Override HARNESS_MODEL")
        command.add_argument(
            "--test-python", type=Path, help="Python interpreter for the target project's pytest"
        )
        if name == "run":
            command.add_argument("prompt", help="Coding task to complete")
    return root


def run_turn(agent, settings: Settings, thread_id: str, prompt: str, tracing: Tracing) -> str:
    handler = tracing.handler()
    config = {
        "configurable": {"thread_id": thread_id},
        "callbacks": [handler] if handler else [],
        "run_name": "coding-harness",
        "metadata": {
            "langfuse_session_id": thread_id,
            "langfuse_tags": ["coding-harness"],
        },
        "recursion_limit": 4 * (settings.max_model_calls + settings.max_tool_calls) + 20,
    }
    final = ""
    try:
        for update in agent.stream(
            {"messages": [{"role": "user", "content": prompt}]},
            config=config,
            stream_mode="updates",
        ):
            for state in update.values():
                if not isinstance(state, dict):
                    continue
                for message in state.get("messages", []):
                    if isinstance(message, ToolMessage):
                        print(f"[tool] {message.name}: {message.status}", file=sys.stderr)
                    elif isinstance(message, AIMessage):
                        for call in message.tool_calls:
                            print(f"[agent] {call['name']}", file=sys.stderr)
                        if not message.tool_calls:
                            final = message.text
        if not final:
            raise RuntimeError("Agent finished without a text answer. Check the saved session.")
        return final
    finally:
        if handler and handler.last_trace_id:
            print(f"[langfuse] trace: {handler.last_trace_id}", file=sys.stderr)


def main(argv: list[str] | None = None) -> None:
    args = parser().parse_args(argv)
    tracing = None
    try:
        options = {}
        if args.no_tracing:
            options["tracing"] = False
        if args.model:
            options["model"] = args.model
        settings = Settings(**options)
        settings.validate_credentials()
        workspace = Workspace(args.workspace)
        test_python = None
        if args.test_python:
            interpreter = args.test_python.expanduser().resolve(strict=True)
            if not interpreter.is_file() or not os.access(interpreter, os.X_OK):
                raise ValueError("--test-python must point to an executable Python interpreter.")
            test_python = str(interpreter)
        session = args.session or str(uuid4())
        if not session.strip():
            raise ValueError("Session label must not be empty.")
        thread_id = checkpoint_id(workspace.root, session)
        print(f"Workspace: {workspace.root}\nSession: {session}", file=sys.stderr)
        tracing = Tracing(settings)
        with open_agent(settings, workspace, test_python) as agent:
            if args.command == "run":
                if not args.prompt.strip():
                    raise ValueError("Prompt must not be empty.")
                print(run_turn(agent, settings, thread_id, args.prompt, tracing))
            else:
                print("Enter a task. Type /exit to leave; Ctrl-D also exits.", file=sys.stderr)
                while True:
                    try:
                        prompt = input("you> ").strip()
                    except EOFError:
                        break
                    if prompt == "/exit":
                        break
                    if prompt:
                        print(run_turn(agent, settings, thread_id, prompt, tracing))
    except KeyboardInterrupt:
        print("\nInterrupted. Completed edits remain in the workspace.", file=sys.stderr)
        raise SystemExit(130) from None
    except Exception as error:
        # SDK errors may contain request bodies or headers. Keep terminal output bounded.
        if isinstance(error, ValueError):
            print(f"Error: {error}", file=sys.stderr)
        else:
            print(
                f"Run failed ({type(error).__name__}). Check the session or trace.",
                file=sys.stderr,
            )
        raise SystemExit(1) from None
    finally:
        if tracing:
            tracing.close()
