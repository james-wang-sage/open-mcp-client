"""
This is the main entry point for the agent.
It defines the workflow graph, state, tools, nodes and edges.
"""

from typing_extensions import Literal, TypedDict, Dict, List, Any, Union, Optional
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command
from copilotkit import CopilotKitState
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import SystemMessage, AIMessage
from utils.logger import mcp_logger
import os
import time
from langchain_ollama.chat_models import ChatOllama
import re

# Define the connection type structures
class StdioConnection(TypedDict):
    command: str
    args: List[str]
    transport: Literal["stdio"]
    enabled: bool

class SSEConnection(TypedDict):
    url: str
    transport: Literal["sse"]
    enabled: bool

# Type for MCP configuration
MCPConfig = Dict[str, Union[StdioConnection, SSEConnection]]

# Refined system prompt
SYSTEM_PROMPT = """You are a helpful assistant specializing in Sage Intacct products. Assume all questions are related to Sage Intacct unless the user explicitly states otherwise.
1. Answer questions based on your knowledge of Sage Intacct.
2. If you need additional information or capabilities to answer (even within the assumed Sage Intacct context), check if any available tools (registered for this session) can help. Use relevant tools when necessary.
3. When asked questions about Intacct objects (models):
   - If you do not have the object schema information in the context, first use the 'listAvailableModels' tool to confirm the object name.
   - Then use the 'getModelDefinition' tool to get the details about the object. When calling 'getModelDefinition', only use the 'name' field and do NOT add a 'type' field or any other field unless explicitly asked to. For example:
     {
       \"name\": \"objects/company-config/employee\"
     }
     Do NOT include 'type', e.g. this is incorrect:
     {
       \"name\": \"objects/general-ledger/journal-entry\",
       \"type\": \"object\"
     }
   - When querying, use the 'executeQuery' tool only after you have the object details.
   - Only use fields that are present in the object details returned by 'getModelDefinition'.
   - Do not guess or use any unknown fields in the query.
   - Anything used in the 'executeQuery' tool must be defined in the schema or tools.
4. If, after checking your knowledge and available tools, you still cannot answer the question (or if the user explicitly stated the question is *not* about Sage Intacct and you lack the knowledge/tools), state that you cannot provide an answer.
5. If the user enters '/skills', list all the tools currently available to you in this session, including their names and descriptions."""

class AgentState(CopilotKitState):
    """
    Here we define the state of the agent

    In this instance, we're inheriting from CopilotKitState, which will bring in
    the CopilotKitState fields. We're also adding a custom field, `mcp_config`,
    which will be used to configure MCP services for the agent.
    """
    # Define mcp_config as an optional field without skipping validation
    mcp_config: Optional[MCPConfig]
    # Add selected_model as an optional field
    selected_model: Optional[str]

# Default MCP configuration to use when no configuration is provided in the state
# Uses relative paths that will work within the project structure
DEFAULT_MCP_CONFIG: MCPConfig = {
    # "math": {
    #     "command": "python",
    #     # Use a relative path that will be resolved based on the current working directory
    #     "args": [os.path.join(os.path.dirname(__file__), "..", "math_server.py")],
    #     "transport": "stdio",
    #     "enabled": True,
    # },
}

# List of supported models
SUPPORTED_MODELS = [
    {"name": "Ollama Qwen3:30B-A3B", "id": "ollama-qwen3", "type": "ollama", "model": "qwen3:30b-a3b", "base_url": "http://localhost:11111"},
    {"name": "OpenAI GPT-4o-mini", "id": "openai-gpt-4o-mini", "type": "openai", "model": "gpt-4o", "openai_api_key_env": "OPENAI_API_KEY"},
    {"name": "OpenAI o4-mini", "id": "openai-o4-mini", "type": "openai", "model": "o4-mini", "openai_api_key_env": "OPENAI_API_KEY"},
]
DEFAULT_MODEL_INDEX = 2

def format_thinking_and_answer(text: str) -> str:
    """
    Formats the response so that the thinking section starts with bold 'Thinking:'
    and the answer section starts with bold 'Answer:'. Removes <think> tags.
    """
    think_match = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
    if think_match:
        thinking = think_match.group(1).strip()
        answer = text.replace(think_match.group(0), "").strip()
        return f"**Thinking:**\n{thinking}\n\n**Answer:**\n{answer}"
    else:
        return text

async def chat_node(state: AgentState, config: RunnableConfig) -> Command[Literal["__end__"]]:
    """
    This is a simplified agent that uses the ReAct agent as a subgraph.
    It handles both chat responses and tool execution in one node.
    """
    start_time = time.time()
    try:
        # Get MCP configuration from state
        state_config = state.get("mcp_config", {})
        # Merge state config with default config, giving priority to state config
        mcp_config = DEFAULT_MCP_CONFIG.copy()
        mcp_config.update(state_config)
        # Filter out disabled servers and remove enabled field from config
        enabled_mcp_config = {}
        for name, server_config in mcp_config.items():
            if server_config.get("enabled", True) is not False:
                filtered_config = server_config.copy()
                filtered_config.pop("enabled", None)
                enabled_mcp_config[name] = filtered_config
        mcp_logger.log_server_selection(enabled_mcp_config, DEFAULT_MCP_CONFIG)
        async with MultiServerMCPClient(enabled_mcp_config) as mcp_client:
            mcp_tools = mcp_client.get_tools()
            last_message = state["messages"][-1] if state.get("messages") else None
            # Handle /models command
            if last_message and isinstance(last_message.content, str) and last_message.content.strip() == '/models':
                models_list = "Here are the supported models (use /model <index> to select):\n\n"
                for idx, model in enumerate(SUPPORTED_MODELS):
                    selected = " (selected)" if state.get("selected_model") == model["id"] or (state.get("selected_model") is None and idx == DEFAULT_MODEL_INDEX) else ""
                    models_list += f"{idx}: {model['name']}{selected}\n"
                ai_response_message = AIMessage(content=models_list)
                current_history = state.get("messages", [])
                updated_messages = current_history + [ai_response_message]
                response_time = (time.time() - start_time) * 1000
                mcp_logger.log_performance(
                    server_name="agent",
                    operation="models_command",
                    duration=response_time,
                    success=True
                )
                return Command(
                    goto=END,
                    update={"messages": updated_messages},
                )
            # Handle /model <index> command
            if last_message and isinstance(last_message.content, str) and last_message.content.strip().startswith('/model'):
                parts = last_message.content.strip().split()
                if len(parts) == 2 and parts[1].isdigit():
                    idx = int(parts[1])
                    if 0 <= idx < len(SUPPORTED_MODELS):
                        selected_model_id = SUPPORTED_MODELS[idx]["id"]
                        state["selected_model"] = selected_model_id
                        confirmation = f"Model changed to: {SUPPORTED_MODELS[idx]['name']} (index {idx})"
                    else:
                        confirmation = f"Invalid model index: {idx}. Use /models to see available models."
                else:
                    confirmation = "Usage: /model <index>. Use /models to see available models."
                ai_response_message = AIMessage(content=confirmation)
                current_history = state.get("messages", [])
                updated_messages = current_history + [ai_response_message]
                response_time = (time.time() - start_time) * 1000
                mcp_logger.log_performance(
                    server_name="agent",
                    operation="model_select_command",
                    duration=response_time,
                    success=True
                )
                return Command(
                    goto=END,
                    update={"messages": updated_messages, "selected_model": state.get("selected_model")},
                )
            # Handle /skills command (existing)
            if last_message and isinstance(last_message.content, str) and last_message.content.strip() == '/skills':
                if mcp_tools:
                    skills_list = "Here are the tools available in this session:\n\n"
                    for tool in mcp_tools:
                        description = getattr(tool, 'description', 'No description available.')
                        skills_list += f"- **{tool.name}**: {description}\n"
                else:
                    skills_list = "There are currently no tools available in this session."
                ai_response_message = AIMessage(content=skills_list)
                current_history = state.get("messages", [])
                updated_messages = current_history + [ai_response_message]
                response_time = (time.time() - start_time) * 1000
                mcp_logger.log_performance(
                    server_name="agent",
                    operation="skills_command",
                    duration=response_time,
                    success=True
                )
                return Command(
                    goto=END,
                    update={"messages": updated_messages},
                )
            # Proceed with the normal agent flow if not a command
            # Determine which model to use
            selected_model_id = state.get("selected_model")
            if selected_model_id is None:
                selected_model_id = SUPPORTED_MODELS[DEFAULT_MODEL_INDEX]["id"]
            model_info = next((m for m in SUPPORTED_MODELS if m["id"] == selected_model_id), SUPPORTED_MODELS[DEFAULT_MODEL_INDEX])
            # Instantiate the model
            if model_info["type"] == "ollama":
                model = ChatOllama(model=model_info["model"], base_url=model_info["base_url"])
            elif model_info["type"] == "openai":
                api_key = os.environ.get(model_info["openai_api_key_env"])
                if not api_key:
                    raise RuntimeError(f"OpenAI API key not found in environment variable {model_info['openai_api_key_env']}")
                # Set temperature=1 for OpenAI models to avoid unsupported value errors
                model = ChatOpenAI(model=model_info["model"], openai_api_key=api_key, temperature=1)
            else:
                raise RuntimeError(f"Unknown model type: {model_info['type']}")
            react_agent = create_react_agent(model, mcp_tools)
            MAX_MESSAGES = 10
            all_messages = state["messages"]
            if len(all_messages) > MAX_MESSAGES:
                conversation_messages = all_messages[-MAX_MESSAGES:]
            else:
                conversation_messages = all_messages
            system_message = SystemMessage(content=SYSTEM_PROMPT)
            agent_input_messages = [system_message] + conversation_messages
            agent_input = {
                "messages": agent_input_messages
            }
            agent_response = await react_agent.ainvoke(agent_input)
            new_messages_from_agent = []
            if "messages" in agent_response:
                input_len = len(agent_input_messages)
                if len(agent_response["messages"]) > input_len:
                     new_messages_from_agent = agent_response["messages"][input_len:]
                elif agent_response["messages"] and agent_response["messages"][-1].type == 'ai':
                    new_messages_from_agent = [agent_response["messages"][-1]]
            current_history_for_update = state.get("messages", [])
            formatted_new_messages = []
            for msg in new_messages_from_agent:
                if getattr(msg, "type", None) == "ai" and hasattr(msg, "content"):
                    msg.content = format_thinking_and_answer(msg.content)
                formatted_new_messages.append(msg)
            updated_messages = current_history_for_update + formatted_new_messages
            response_time = (time.time() - start_time) * 1000
            mcp_logger.log_performance(
                server_name="agent",
                operation="chat_node",
                duration=response_time,
                success=True
            )
            return Command(
                goto=END,
                update={"messages": updated_messages},
            )
    except Exception as e:
        mcp_logger.log_error(
            server_name="agent",
            error_type=type(e).__name__,
            error_message=str(e),
            stack_trace=None
        )
        raise

# Define the workflow graph with only a chat node
workflow = StateGraph(AgentState)
workflow.add_node("chat_node", chat_node)
workflow.set_entry_point("chat_node")

# Compile the workflow graph
graph = workflow.compile(MemorySaver())