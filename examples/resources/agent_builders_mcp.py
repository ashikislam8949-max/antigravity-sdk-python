# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Stdio MCP server for the Agent Builders Bot.

Named tools are advertised to the agent the same way a Grok bot MCP
connection uses ``advertiseTools: true``: the client sees
``list_sdk_patterns``, ``draft_agent_spec``, ``get_agent_spec``, and
``write_agent_scaffold`` by those exact names.

To run standalone:
  python agent_builders_mcp.py --transport=stdio --scaffold-dir=./agent_scaffolds
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from typing import Any

from mcp.server.fastmcp import server

_SAFE_FILENAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*\.py$")

_SDK_PATTERNS = [
    {
        "id": "hello_world",
        "path": "examples/getting_started/hello_world.py",
        "summary": "Minimal Agent + LocalAgentConfig chat turn.",
    },
    {
        "id": "persona_config",
        "path": "examples/getting_started/persona_config.py",
        "summary": "TemplatedSystemInstructions for identity and sections.",
    },
    {
        "id": "custom_tools",
        "path": "examples/getting_started/custom_tools.py",
        "summary": "Python callables as tools with ToolContext state.",
    },
    {
        "id": "policies",
        "path": "examples/getting_started/policies.py",
        "summary": "Deny-by-default policies, allowlists, ask_user.",
    },
    {
        "id": "agent_skills",
        "path": "examples/getting_started/agent_skills.py",
        "summary": "Load domain skills via skills_paths / SKILL.md.",
    },
    {
        "id": "subagents",
        "path": "examples/getting_started/subagents.py",
        "summary": "Static and dynamic subagent delegation.",
    },
    {
        "id": "structured_output",
        "path": "examples/getting_started/structured_output.py",
        "summary": "Pydantic response_schema for typed JSON output.",
    },
    {
        "id": "persistence",
        "path": "examples/getting_started/persistence.py",
        "summary": "Resume conversations with conversation_id + save_dir.",
    },
    {
        "id": "hooks",
        "path": "examples/getting_started/hooks.py",
        "summary": "Lifecycle hooks (pre/post turn, tool events).",
    },
    {
        "id": "interactive_cli",
        "path": "examples/deep_dives/interactive_cli.py",
        "summary": "Streaming REPL with MCP, policies, and telemetry.",
    },
]

# Session state lives in this process, which the stdio client keeps alive
# for the agent session (same role as a Grok bot connection's tool process).
_STATE: dict[str, Any] = {}
_SCAFFOLD_DIR = ""


def configure_scaffold_dir(path: str) -> None:
  """Sets the directory ``write_agent_scaffold`` is allowed to write into."""
  global _SCAFFOLD_DIR
  _SCAFFOLD_DIR = os.path.abspath(path) if path else ""


def list_sdk_patterns() -> str:
  """Lists known Google Antigravity SDK example patterns to reuse.

  Returns:
    A JSON string catalog of pattern ids, paths, and short summaries.
  """
  return json.dumps({"patterns": _SDK_PATTERNS}, indent=2)


def draft_agent_spec(
    name: str,
    purpose: str,
    tools: str,
    skills: str,
    notes: str,
) -> str:
  """Stores a structured agent design draft for this MCP session.

  Args:
    name: Short slug or display name for the agent.
    purpose: One-sentence description of what the agent does.
    tools: Comma-separated tool names or brief tool descriptions.
    skills: Comma-separated skill names or paths (may be empty).
    notes: Extra design notes (policies, model, channels, etc.).

  Returns:
    A confirmation string including the saved JSON draft.
  """
  draft = {
      "name": name.strip(),
      "purpose": purpose.strip(),
      "tools": [t.strip() for t in tools.split(",") if t.strip()],
      "skills": [s.strip() for s in skills.split(",") if s.strip()],
      "notes": notes.strip(),
  }
  _STATE["agent_spec"] = draft
  return "Saved agent spec draft:\n" + json.dumps(draft, indent=2)


def get_agent_spec() -> str:
  """Returns the current agent design draft for this MCP session.

  Returns:
    JSON for the saved draft, or a message if none exists yet.
  """
  draft = _STATE.get("agent_spec")
  if not draft:
    return "No agent spec drafted yet. Call draft_agent_spec first."
  return json.dumps(draft, indent=2)


def write_agent_scaffold(filename: str, code: str) -> str:
  """Writes a Python agent scaffold under the configured output directory.

  Only bare ``*.py`` filenames are allowed (no path separators).

  Args:
    filename: Destination file name ending in ``.py`` (e.g. ``review_agent.py``).
    code: Full Python source to write.

  Returns:
    Absolute path written, or an error message if the filename is rejected.
  """
  if not _SCAFFOLD_DIR:
    return "Error: scaffold directory is not configured."

  name = filename.strip()
  if (
      not name
      or name != os.path.basename(name)
      or os.sep in name
      or "/" in name
      or "\\" in name
      or ".." in name
      or not _SAFE_FILENAME.match(name)
  ):
    return (
        "Error: filename must be a simple *.py name using letters, digits,"
        " underscores, or hyphens (e.g. my_agent.py)."
    )

  os.makedirs(_SCAFFOLD_DIR, exist_ok=True)
  dest = os.path.abspath(os.path.join(_SCAFFOLD_DIR, name))
  scaffold_root = os.path.abspath(_SCAFFOLD_DIR)
  if not dest.startswith(scaffold_root + os.sep):
    return "Error: refused to write outside the scaffold directory."
  with open(dest, "w", encoding="utf-8") as f:
    f.write(code)
  return f"Wrote scaffold to {dest}"


def create_server() -> server.FastMCP:
  """Creates the Agent Builders FastMCP server with named tools."""
  mcp = server.FastMCP("Agent Builders")
  mcp.tool()(list_sdk_patterns)
  mcp.tool()(draft_agent_spec)
  mcp.tool()(get_agent_spec)
  mcp.tool()(write_agent_scaffold)
  return mcp


def main(argv: list[str] | None = None) -> None:
  """Runs the MCP server on stdio (default), SSE, or streamable HTTP."""
  parser = argparse.ArgumentParser(
      description="Agent Builders MCP server."
  )
  parser.add_argument(
      "--scaffold-dir",
      default=os.environ.get("AGENT_BUILDERS_SCAFFOLD_DIR", ""),
      help="Directory write_agent_scaffold may write *.py files into.",
  )
  parser.add_argument(
      "--transport",
      choices=["stdio", "sse", "streamable-http"],
      default="stdio",
      help="MCP transport. The Agent Builders Bot uses stdio.",
  )
  parser.add_argument(
      "--port", type=int, default=8000, help="Port for sse / http transports."
  )
  args = parser.parse_args(argv)
  configure_scaffold_dir(args.scaffold_dir)
  mcp = create_server()
  mcp.settings.host = "0.0.0.0"
  mcp.settings.port = args.port
  mcp.run(transport=args.transport)


if __name__ == "__main__":
  main(sys.argv[1:])
