# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Project Overview

This is an open-source MCP (Model Control Protocol) client built with CopilotKit that enables orchestration of multiple AI agents and skills. The application serves as a unified interface for different Copilot capabilities, allowing dynamic skill activation based on user profile and company configuration.

## Architecture

The codebase is split into two main parts:

### Frontend (`/app` folder)
- **Next.js 15** application with Turbopack
- **CopilotKit** for UI and state synchronization 
- **Tailwind CSS** with custom components
- **TypeScript** with React 19

### Agent (`/agent` folder) 
- **LangGraph** agent that connects to MCP servers and calls their tools
- **Python Poetry** for dependency management
- **FastMCP** for creating MCP servers
- **Multiple LLM support** (OpenAI, Ollama)

## Development Commands

### Quick Start
```bash
# Set up environment variables first
touch .env
# Add LANGSMITH_API_KEY=lsv2_... to root .env

cd agent && touch .env
# Add OPENAI_API_KEY, LANGSMITH_API_KEY, OPENROUTER_API_KEY to agent/.env
```

### Development (Recommended - Separate Terminals)
```bash
# Terminal 1 - Frontend
pnpm run dev-frontend

# Terminal 2 - Agent
pnpm run dev-agent
```

### Development (Single Command)
```bash
pnpm run dev
```

### Testing and Build
```bash
# Frontend linting
pnpm lint

# Frontend build
pnpm build
pnpm start

# Agent (inside agent/ directory)
poetry install
poetry run langgraph dev --host localhost --port 8123 --no-browser
```

## Core Components

### State Management
- `AgentState` extends `CopilotKitState` with MCP configuration
- `mcp_config`: Dictionary of server configurations (stdio/sse)
- `selected_model`: Currently selected LLM model
- State persists in localStorage via `useLocalStorage` hook

### MCP Server Configuration
Two transport types supported:
- **stdio**: Command-based servers (`command`, `args`, `transport: "stdio"`)
- **sse**: HTTP endpoint servers (`url`, `transport: "sse"`)

### Agent Commands
- `/models` - List available LLM models
- `/model <index>` - Switch to specific model
- `/skills` - List available MCP tools/capabilities

## Key Files

### Frontend
- `app/page.tsx` - Main UI with chat interface and MCP configuration
- `app/components/MCPConfigForm.tsx` - MCP server management interface  
- `app/components/CopilotActionHandler.tsx` - Custom action rendering
- `app/api/copilotkit/route.ts` - CopilotKit runtime configuration

### Agent
- `agent/sample_agent/agent.py` - Main LangGraph workflow with ReAct pattern
- `agent/langgraph.json` - LangGraph configuration
- `agent/pyproject.toml` - Poetry dependencies
- `agent/math_server.py` - Example MCP server implementation

## Multi-Model Support

Supported models configured in `SUPPORTED_MODELS`:
- Ollama models (local)
- OpenAI models (GPT-4o, o4-mini)
- Configurable via environment variables

Default model index: 2 (OpenAI o4-mini)

## MCP Integration

- Uses `MultiServerMCPClient` from `langchain-mcp-adapters`
- Servers can be enabled/disabled dynamically
- Configuration merged from defaults and state
- Tools automatically registered with LangGraph ReAct agent

## Error Handling & Logging

- Custom MCP logger in `utils/logger.py`
- Performance tracking for operations
- Error logging with server context
- Graceful fallbacks for missing configurations

## Environment Variables

### Root `.env`
```
LANGSMITH_API_KEY=lsv2_...
```

### Agent `.env` 
```
OPENAI_API_KEY=sk-...
LANGSMITH_API_KEY=lsv2_...
OPENROUTER_API_KEY=sk-or-...
```

## Package Management

- **Frontend**: pnpm (specified in package.json)
- **Agent**: Poetry (Python dependencies)

## Development Notes

- Frontend runs on port 3000 by default
- Agent runs on port 8123 (LangGraph dev server)
- Uses Turbopack for faster development builds
- State synchronization between frontend and agent via CopilotKit
- MCP servers can be added dynamically without code changes

## Demo Scenarios

The `demo.txt` contains example workflows for:
- Skill orchestration (Help Search, Tavily, Weather)
- Model and Query operations
- Accounts Payable operations
- Dynamic skill activation based on context
