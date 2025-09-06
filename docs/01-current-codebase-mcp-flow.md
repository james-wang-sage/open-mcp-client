# Current Codebase: MCP Tools and Conversation Flow

## Overview

This document explains how the current open-mcp-client codebase handles MCP tool preparation, context management, and conversation flow using LangGraph + LangChain architecture.

## Architecture Components

### Core Technologies
- **LangGraph**: Workflow orchestration and state management
- **LangChain**: LLM integration and ReAct agent pattern
- **CopilotKit**: Frontend-backend state synchronization
- **MultiServerMCPClient**: MCP protocol implementation

## 1. MCP Tools Preparation

### Connection Setup (Per Request)
```python
# agent/sample_agent/agent.py:125
async with MultiServerMCPClient(enabled_mcp_config) as mcp_client:
    mcp_tools = mcp_client.get_tools()
```

**Process Flow:**
1. **Configuration Merging** (Lines 112-116)
   ```python
   state_config = state.get("mcp_config", {})
   mcp_config = DEFAULT_MCP_CONFIG.copy()
   mcp_config.update(state_config)
   ```

2. **Server Filtering** (Lines 118-123)
   ```python
   enabled_mcp_config = {}
   for name, server_config in mcp_config.items():
       if server_config.get("enabled", True) is not False:
           filtered_config = server_config.copy()
           filtered_config.pop("enabled", None)
           enabled_mcp_config[name] = filtered_config
   ```

3. **Tool Discovery** (Line 126)
   - `MultiServerMCPClient` connects to all enabled servers
   - `get_tools()` retrieves available tools from each server
   - Tools are registered for the ReAct agent

### Tool Registration Pattern
```python
# Line 215
react_agent = create_react_agent(model, mcp_tools)
```

**Current Issues:**
- ❌ **Connection per request**: New connections for every user message
- ❌ **No tool caching**: Tools discovered fresh each time
- ❌ **No optimization**: All tools loaded regardless of relevance

## 2. Context Management

### Conversation History (Manual)
```python
# Lines 216-223
MAX_MESSAGES = 10
all_messages = state["messages"]
if len(all_messages) > MAX_MESSAGES:
    conversation_messages = all_messages[-MAX_MESSAGES:]
else:
    conversation_messages = all_messages
system_message = SystemMessage(content=state.get("system_prompt", SYSTEM_PROMPT))
agent_input_messages = [system_message] + conversation_messages
```

**Context Structure:**
- **System prompt**: Sage Intacct specialized instructions
- **Last 10 messages**: Manual history truncation
- **Current message**: User's latest query
- **MCP tools**: Available capabilities for this session

### State Persistence
```python
# Frontend: MCPConfigForm.tsx
const { state: agentState, setState: setAgentState } = useCoAgent<AgentState>({
    name: "sample_agent",
    initialState: {
        mcp_config: savedConfigs,
        system_prompt: DEFAULT_SYSTEM_PROMPT,
    },
});
```

**State Components:**
- `mcp_config`: Server configurations and enabled status
- `system_prompt`: Customizable system instructions
- `selected_model`: Currently active LLM model
- `messages`: Full conversation history

## 3. Complete Conversation Flow

### User Input Processing
```
User types: "find unpaid bills for PG&E"
    ↓
Frontend (CopilotKit) captures message
    ↓
API call to /api/copilotkit
    ↓
LangGraph workflow triggered: chat_node()
```

### Agent Execution Flow

#### Step 1: Context Preparation
```python
# chat_node function execution
state_config = state.get("mcp_config", {})  # Get MCP servers config
mcp_config = DEFAULT_MCP_CONFIG.copy()      # Merge with defaults
enabled_mcp_config = filter_enabled_servers() # Only active servers
```

#### Step 2: MCP Client Connection
```python
async with MultiServerMCPClient(enabled_mcp_config) as mcp_client:
    mcp_tools = mcp_client.get_tools()  # Discover available tools
```

#### Step 3: Model Selection & Agent Creation
```python
selected_model_id = state.get("selected_model") or DEFAULT_MODEL_INDEX
model = create_model_instance(selected_model_id)  # OpenAI, Ollama, etc.
react_agent = create_react_agent(model, mcp_tools)
```

#### Step 4: History Management
```python
conversation_messages = all_messages[-MAX_MESSAGES:]  # Last 10 only
system_message = SystemMessage(content=system_prompt)
agent_input_messages = [system_message] + conversation_messages
```

#### Step 5: ReAct Agent Execution
```python
agent_response = await react_agent.ainvoke(agent_input)
```

**Inside ReAct Loop:**
1. **LLM Reasoning**: Analyzes user query
2. **Tool Selection**: Decides which MCP tools to use
3. **Tool Execution**: Callbacks to MCP servers via client
4. **Result Processing**: Analyzes tool responses
5. **Response Generation**: Creates final answer

#### Step 6: Response Processing
```python
new_messages_from_agent = extract_new_messages(agent_response)
formatted_messages = format_thinking_and_answer(new_messages_from_agent)
updated_messages = current_history + formatted_messages
```

#### Step 7: State Update
```python
return Command(
    goto=END,
    update={"messages": updated_messages}
)
```

## 4. Special Commands Handling

### Built-in Commands
- **`/models`**: List available LLM models
- **`/model <index>`**: Switch active model
- **`/skills`**: Display available MCP tools

These bypass normal ReAct flow for quick responses.

## 5. Current Limitations

### Performance Issues
1. **Connection Overhead**: New MCP connections per request
2. **Tool Rediscovery**: No caching of available tools
3. **Context Inefficiency**: Manual message history management
4. **No Smart Filtering**: All tools loaded regardless of query relevance

### Scalability Concerns
1. **Memory Usage**: Full conversation history in state
2. **Latency**: Connection setup delays
3. **Resource Waste**: Unused tool connections maintained

### Architecture Limitations
1. **Tight Coupling**: MCP client lifecycle tied to request lifecycle
2. **No Optimization**: No intelligent tool selection
3. **Static Configuration**: Limited dynamic adaptation

## 6. Key Files and Components

### Core Agent Logic
- `agent/sample_agent/agent.py`: Main workflow implementation
- `agent/langgraph.json`: LangGraph configuration
- `agent/utils/logger.py`: Performance and error logging

### Frontend Integration
- `app/components/MCPConfigForm.tsx`: Server configuration UI
- `app/components/CopilotActionHandler.tsx`: Custom action rendering
- `app/api/copilotkit/route.ts`: CopilotKit runtime endpoint

### State Management
- `app/hooks/useLocalStorage.ts`: Persistent storage utility
- CopilotKit handles frontend-backend state synchronization

## Summary

The current codebase provides a functional MCP integration with basic conversation capabilities, but uses a **request-per-connection** pattern that creates performance bottlenecks and missed optimization opportunities. The architecture is well-structured for development but needs optimization for production scale.

**Next Steps**: Modern OpenAI API features and optimized MCP client patterns can significantly improve performance and user experience.
