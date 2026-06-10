# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

An open-source MCP (Model Context Protocol) client built with CopilotKit and LangGraph. The Next.js frontend lets users configure MCP servers dynamically; a Python LangGraph agent connects to those servers and exposes their tools to an LLM via a ReAct loop. The agent's system prompt specializes it for Sage Intacct questions.

## Commands

```bash
pnpm run dev            # Run frontend + agent together (concurrently)
pnpm run dev-frontend   # pnpm i && next dev --turbopack (port 3000)
pnpm run dev-agent      # cd agent && poetry install && poetry run langgraph dev --host localhost --port 8123 --no-browser
pnpm lint               # next lint
pnpm build              # next build
pnpm start              # next start (production)
```

- Frontend: pnpm 11.5.3 (pinned via `packageManager`). Agent: Poetry, Python 3.12 (per `agent/langgraph.json`).
- There is no test infrastructure (no test scripts or test files).

## Environment Variables

- Root `.env`: `LANGSMITH_API_KEY`
- `agent/.env`: `OPENAI_API_KEY`, `LANGSMITH_API_KEY`, `OPENROUTER_API_KEY`
- `AGENT_DEPLOYMENT_URL` — agent URL used by `app/api/copilotkit/route.ts` (defaults to `http://localhost:8123`)

## Architecture

Two-part application (single repo, not a workspace monorepo):

```
app/ (Next.js 15 + React 19 + CopilotKit)          agent/ (Python LangGraph)
  page.tsx — chat UI + MCP config                    sample_agent/agent.py — entire workflow
  components/MCPConfigForm.tsx — server management   langgraph.json — registers sample_agent graph
  api/copilotkit/route.ts — CopilotKit runtime       utils/logger.py — MCP perf/error logging
```

**Request flow:** Chat UI → POST `/api/copilotkit` → `CopilotRuntime` with `langGraphPlatformEndpoint` → LangGraph dev server on `:8123` → `chat_node` in `agent/sample_agent/agent.py`.

**Agent name coupling:** The string `sample_agent` must match in three places — `agent/langgraph.json` (graph registration), `app/api/copilotkit/route.ts` (agents list), and `app/components/MCPConfigForm.tsx` (`useCoAgent` hook). Renaming the agent requires updating all three.

**State sync:** `AgentState` (agent.py) extends `CopilotKitState` with `mcp_config` and `selected_model`. The frontend reads/writes this state through CopilotKit's shared-state mechanism (`useCoAgent` in MCPConfigForm), and persists configs in browser localStorage via `app/hooks/useLocalStorage.ts`. Changing the state shape means updating both the Python TypedDicts and the frontend's `AgentState` type.

**Agent internals (`agent/sample_agent/agent.py`):**
- The graph has a single node, `chat_node`. On every turn it opens a fresh `MultiServerMCPClient` from the enabled MCP servers in state, fetches their tools, and runs them through a `create_react_agent` subgraph.
- MCP server configs are merged: `DEFAULT_MCP_CONFIG` ← state `mcp_config`, then servers with `enabled: false` are filtered out and the `enabled` key stripped before being passed to the MCP client.
- Two transport types: `stdio` (`command`, `args`) and `sse` (`url`).
- Slash commands are intercepted in `chat_node` before reaching the LLM: `/models` (list), `/model <index>` (switch), `/skills` (list MCP tools).
- Models live in `SUPPORTED_MODELS` (Ollama local + OpenAI); `DEFAULT_MODEL_INDEX = 2` (OpenAI o4-mini). Add models by appending to this list.
- Conversation history sent to the LLM is truncated to the last 10 messages (`MAX_MESSAGES`).
- The system prompt can be overridden per-session via `system_prompt` in state (editable in the UI); otherwise `SYSTEM_PROMPT` (Sage Intacct-focused) is used.
