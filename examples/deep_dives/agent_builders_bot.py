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

This deep-dive example combines persona configuration, skill loading, a
stdio MCP server, safety policies, and a streaming CLI REPL. The MCP
server advertises named tools (the same idea as a Grok bot connection
with ``advertiseTools: true``): ``list_sdk_patterns``,
``draft_agent_spec``, ``get_agent_spec``, and ``write_agent_scaffold``.

The bot:
1. Interviews the user about the agent they want to build.
2. Consults the google-antigravity-sdk skill for SDK patterns.
3. Drafts a structured agent spec through the MCP server.
4. Writes a starter ``.py`` scaffold under ``./agent_scaffolds/``.

To run:
  export GEMINI_API_KEY="your_api_key_here"
  python agent_builders_bot.py

Criteria for correct script performance:
  1. The script exits cleanly with return code 0 (no unhandled exceptions).
  2. The banner prints the MCP server path and the interactive loop accepts
     user input.
  3. The agent can call the advertised MCP tools by name.
  4. write_agent_scaffold only writes ``.py`` files under the scaffold directory.
  5. Typing ``exit`` or ``quit`` (or Ctrl+C) ends the session cleanly.
"""

from __future__ import annotations

import asyncio
import os
import sys

from google.antigravity import types
from google.antigravity import Agent, LocalAgentConfig
from google.antigravity.hooks import policy
from google.antigravity.utils.interactive import AskQuestionHook, async_input

# Exact names advertised by examples/resources/agent_builders_mcp.py.
_MCP_TOOLS = (
    "list_sdk_patterns",
    "draft_agent_spec",
    "get_agent_spec",
    "write_agent_scaffold",
)


def _skill_path() -> str:
  """Resolves the path to the google-antigravity-sdk skill directory."""
  script_dir = os.path.dirname(os.path.abspath(__file__))
  return os.path.abspath(
      os.path.join(script_dir, "../../skills/google-antigravity-sdk")
  )


def _mcp_server_path() -> str:
  """Resolves the Agent Builders stdio MCP server script."""
  script_dir = os.path.dirname(os.path.abspath(__file__))
  return os.path.abspath(
      os.path.join(script_dir, "../resources/agent_builders_mcp.py")
  )


def _build_system_instructions(
    scaffold_dir: str, skill_path: str
) -> types.TemplatedSystemInstructions:
  """Builds templated system instructions for the Agent Builders Bot."""
  identity = (
      "You are Agent Builders Bot, an expert guide for designing and"
      " scaffolding Google Antigravity SDK agents.\n"
      "Your job is to interview the user briefly, recommend SDK patterns,"
      " draft a clear agent spec, and write a runnable starter Python file.\n"
      "Your design tools come from the Agent Builders MCP server. Call them"
      " by their advertised names."
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
  skill_path = _skill_path()
  mcp_path = _mcp_server_path()
  scaffold_dir = os.path.abspath(os.path.join(os.getcwd(), "agent_scaffolds"))
  os.makedirs(scaffold_dir, exist_ok=True)

  if not os.environ.get("GEMINI_API_KEY"):
    print(
        "Warning: GEMINI_API_KEY is not set. Export it before chatting, e.g.\n"
        '  export GEMINI_API_KEY="your_api_key_here"\n'
    )

  print("=== Agent Builders Bot ===")
  print(f"Skill path:     {skill_path}")
  print(f"MCP server:     {mcp_path}")
  print(f"MCP tools:      {', '.join(_MCP_TOOLS)}")
  print(f"Scaffold dir:   {scaffold_dir}")
  print("Describe the agent you want to build.\n")

  # Stdio MCP connection. Tool names stay visible to the model, matching a
  # Grok bot defineConnection({ command, args, advertiseTools: true }).
  mcp_server = types.McpStdioServer(
      command=sys.executable,
      args=[
          mcp_path,
          "--transport=stdio",
          f"--scaffold-dir={scaffold_dir}",
      ],
  )

  config = LocalAgentConfig(
      system_instructions=_build_system_instructions(scaffold_dir, skill_path),
      skills_paths=[skill_path],
      mcp_servers=[mcp_server],
      workspaces=[scaffold_dir, skill_path],
      capabilities=types.CapabilitiesConfig(
          enabled_tools=[
              *types.BuiltinTools.read_only(),
              types.BuiltinTools.ASK_QUESTION,
          ],
      ),
      policies=[
          policy.deny_all(),
          *[policy.allow(name) for name in _MCP_TOOLS],
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
