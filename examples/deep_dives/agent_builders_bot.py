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

"""Interactive Agent Builders Bot for Google Antigravity SDK.

This deep-dive example combines persona configuration, skill loading, custom
tools, safety policies, and a streaming CLI REPL into a single assistant that
helps users design and scaffold Antigravity agents.

The bot:
1. Interviews the user about the agent they want to build.
2. Consults the google-antigravity-sdk skill for SDK patterns.
3. Drafts a structured agent spec via tools (stored in ToolContext state).
4. Writes a starter ``.py`` scaffold under ``./agent_scaffolds/``.

To run:
  export GEMINI_API_KEY="your_api_key_here"
  python agent_builders_bot.py

Criteria for correct script performance:
  1. The script exits cleanly with return code 0 (no unhandled exceptions).
  2. The banner prints and the interactive loop accepts user input.
  3. The agent can call list_sdk_patterns / draft_agent_spec / get_agent_spec.
  4. write_agent_scaffold only writes ``.py`` files under the scaffold directory.
  5. Typing ``exit`` or ``quit`` (or Ctrl+C) ends the session cleanly.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import sys

from google.antigravity import types
from google.antigravity import Agent, LocalAgentConfig, ToolContext
from google.antigravity.hooks import policy
from google.antigravity.utils.interactive import AskQuestionHook, async_input

# Directory where scaffolds are written. Set in main() before the agent starts.
_SCAFFOLD_DIR: str = ""

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

_SAFE_FILENAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*\.py$")


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
    ctx: ToolContext,
) -> str:
  """Stores or updates a structured agent design draft in session state.

  Args:
    name: Short slug or display name for the agent.
    purpose: One-sentence description of what the agent does.
    tools: Comma-separated tool names or brief tool descriptions.
    skills: Comma-separated skill names or paths (may be empty).
    notes: Extra design notes (policies, model, channels, etc.).
    ctx: Injected tool context for per-conversation state.

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
  ctx.set_state("agent_spec", draft)
  return "Saved agent spec draft:\n" + json.dumps(draft, indent=2)


def get_agent_spec(ctx: ToolContext) -> str:
  """Returns the current agent design draft from session state.

  Args:
    ctx: Injected tool context for per-conversation state.

  Returns:
    JSON for the saved draft, or a message if none exists yet.
  """
  draft = ctx.get_state("agent_spec")
  if not draft:
    return "No agent spec drafted yet. Call draft_agent_spec first."
  return json.dumps(draft, indent=2)


def write_agent_scaffold(filename: str, code: str) -> str:
  """Writes a Python agent scaffold under the configured output directory.

  Only bare ``*.py`` filenames are allowed (no path separators). The file is
  always created inside the scaffold directory prepared at startup.

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


def _skill_path() -> str:
  """Resolves the path to the google-antigravity-sdk skill directory."""
  script_dir = os.path.dirname(os.path.abspath(__file__))
  return os.path.abspath(
      os.path.join(script_dir, "../../skills/google-antigravity-sdk")
  )


def _build_system_instructions(
    scaffold_dir: str, skill_path: str
) -> types.TemplatedSystemInstructions:
  """Builds templated system instructions for the Agent Builders Bot."""
  identity = (
      "You are Agent Builders Bot, an expert guide for designing and"
      " scaffolding Google Antigravity SDK agents.\n"
      "Your job is to interview the user briefly, recommend SDK patterns,"
      " draft a clear agent spec, and write a runnable starter Python file."
  )

  interview_flow = types.SystemInstructionSection(
      title="interview_flow",
      content=(
          "1. Ask 1-3 short questions about purpose, tools/APIs, and safety"
          " needs before writing code.\n"
          "2. Call `list_sdk_patterns` when recommending examples to reuse.\n"
          "3. Call `draft_agent_spec` once you have a coherent design, then"
          " `get_agent_spec` if you need to recall it.\n"
          "4. When the user is ready, call `write_agent_scaffold` with a"
          " complete, copy-pasteable Python starter that uses"
          " `Agent` and `LocalAgentConfig`.\n"
          "5. Prefer `TemplatedSystemInstructions` over full custom system"
          " prompts unless the user explicitly needs a full override.\n"
          "6. Consult the google-antigravity-sdk skill (view its SKILL.md and"
          f" references under `{skill_path}`) when unsure about APIs."
      ),
  )

  scaffolding_rules = types.SystemInstructionSection(
      title="scaffolding_rules",
      content=(
          f"- Scaffolds must be written only via `write_agent_scaffold` into"
          f" `{scaffold_dir}`.\n"
          "- Generated files must be valid Python with an asyncio main entry"
          " point and a clear module docstring.\n"
          "- Include placeholder comments for tools, policies, and skills the"
          " user still needs to fill in.\n"
          "- Do not invent SDK APIs; stick to patterns from the skill and the"
          " listed examples."
      ),
  )

  safety = types.SystemInstructionSection(
      title="safety",
      content=(
          "- You cannot run shell commands or edit arbitrary files.\n"
          "- Keep recommendations practical and scoped; do not over-engineer.\n"
          "- If requirements are unclear, ask before scaffolding."
      ),
  )

  return types.TemplatedSystemInstructions(
      identity=identity,
      sections=[interview_flow, scaffolding_rules, safety],
  )


async def run_repl(agent: Agent) -> None:
  """Runs the streaming interactive chat loop."""
  print("Type your message and press Enter • exit / quit / Ctrl+C to leave\n")
  while True:
    try:
      user_input = await async_input("\n→ ")
      user_input = user_input.strip()
      if not user_input:
        continue
      if user_input.lower() in ("exit", "quit"):
        print("\nGoodbye!")
        break

      response = await agent.chat(user_input)
      async for chunk in response:
        sys.stdout.write(chunk)
        sys.stdout.flush()
      print()
    except (KeyboardInterrupt, asyncio.CancelledError, EOFError):
      print("\nGoodbye!")
      break


async def main() -> None:
  """Configures and starts the Agent Builders Bot CLI."""
  global _SCAFFOLD_DIR

  skill_path = _skill_path()
  _SCAFFOLD_DIR = os.path.abspath(os.path.join(os.getcwd(), "agent_scaffolds"))
  os.makedirs(_SCAFFOLD_DIR, exist_ok=True)

  if not os.environ.get("GEMINI_API_KEY"):
    print(
        "Warning: GEMINI_API_KEY is not set. Export it before chatting, e.g.\n"
        '  export GEMINI_API_KEY="your_api_key_here"\n'
    )

  print("=== Agent Builders Bot ===")
  print(f"Skill path:     {skill_path}")
  print(f"Scaffold dir:   {_SCAFFOLD_DIR}")
  print("Describe the agent you want to build.\n")

  custom_tools = [
      list_sdk_patterns,
      draft_agent_spec,
      get_agent_spec,
      write_agent_scaffold,
  ]

  config = LocalAgentConfig(
      system_instructions=_build_system_instructions(_SCAFFOLD_DIR, skill_path),
      skills_paths=[skill_path],
      tools=custom_tools,
      workspaces=[_SCAFFOLD_DIR, skill_path],
      capabilities=types.CapabilitiesConfig(
          enabled_tools=[
              *types.BuiltinTools.read_only(),
              types.BuiltinTools.ASK_QUESTION,
          ],
      ),
      policies=[
          policy.deny_all(),
          policy.allow(list_sdk_patterns.__name__),
          policy.allow(draft_agent_spec.__name__),
          policy.allow(get_agent_spec.__name__),
          policy.allow(write_agent_scaffold.__name__),
          policy.allow(types.BuiltinTools.VIEW_FILE.value),
          policy.allow(types.BuiltinTools.LIST_DIR.value),
          policy.allow(types.BuiltinTools.SEARCH_DIR.value),
          policy.allow(types.BuiltinTools.FIND_FILE.value),
          policy.allow(types.BuiltinTools.FINISH.value),
          policy.allow(types.BuiltinTools.ASK_QUESTION.value),
      ],
      hooks=[AskQuestionHook()],
  )

  async with Agent(config) as agent:
    await run_repl(agent)


if __name__ == "__main__":
  asyncio.run(main())
