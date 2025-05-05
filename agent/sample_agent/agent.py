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
3. If, after checking your knowledge and available tools, you still cannot answer the question (or if the user explicitly stated the question is *not* about Sage Intacct and you lack the knowledge/tools), state that you cannot provide an answer.
4. If the user enters '/skills', list all the tools currently available to you in this session, including their names and descriptions."""

class AgentState(CopilotKitState):
    """
    Here we define the state of the agent

    In this instance, we're inheriting from CopilotKitState, which will bring in
    the CopilotKitState fields. We're also adding a custom field, `mcp_config`,
    which will be used to configure MCP services for the agent.
    """
    # Define mcp_config as an optional field without skipping validation
    mcp_config: Optional[MCPConfig]

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
            # Explicitly check if enabled is False, default to True if not specified
            if server_config.get("enabled", True) is not False:
                # Create a copy of the config without the enabled field
                filtered_config = server_config.copy()
                filtered_config.pop("enabled", None)
                enabled_mcp_config[name] = filtered_config

        # 记录服务器选择信息
        mcp_logger.log_server_selection(enabled_mcp_config, DEFAULT_MCP_CONFIG)
        
        # Set up the MCP client and tools using only enabled servers
        async with MultiServerMCPClient(enabled_mcp_config) as mcp_client:
            # Get the tools
            mcp_tools = mcp_client.get_tools()
            
            # Check if the last message is the /skills command
            last_message = state["messages"][-1] if state.get("messages") else None
            if last_message and isinstance(last_message.content, str) and last_message.content.strip() == '/skills':
                if mcp_tools:
                    skills_list = "Here are the tools available in this session:\n\n"
                    for tool in mcp_tools:
                        # Ensure description exists, provide fallback if not
                        description = getattr(tool, 'description', 'No description available.')
                        skills_list += f"- **{tool.name}**: {description}\n"
                else:
                    skills_list = "There are currently no tools available in this session."

                # Create the AI response message
                ai_response_message = AIMessage(content=skills_list)

                # Append the AI response to the full history
                # Ensure state["messages"] is treated as a list
                current_history = state.get("messages", [])
                updated_messages = current_history + [ai_response_message]

                # Log performance (optional for this simple command)
                response_time = (time.time() - start_time) * 1000
                mcp_logger.log_performance(
                    server_name="agent",
                    operation="skills_command", # Different operation name
                    duration=response_time,
                    success=True
                )

                # End the graph with the updated messages
                return Command(
                    goto=END,
                    update={"messages": updated_messages},
                )
            else:
                # Proceed with the normal agent flow if not /skills
                # Create the react agent with optimized configuration
                model = ChatOllama(model="qwen3:30b-a3b", base_url="http://localhost:11111")
                react_agent = create_react_agent(model, mcp_tools)
                
                # 截断消息历史，只保留最后10条消息
                MAX_MESSAGES = 10
                all_messages = state["messages"] # Keep original full history reference
                if len(all_messages) > MAX_MESSAGES:
                    # Keep the last MAX_MESSAGES user/assistant messages for context
                    conversation_messages = all_messages[-MAX_MESSAGES:]
                else:
                    conversation_messages = all_messages

                # Prepare messages for the react agent, adding the system prompt
                system_message = SystemMessage(content=SYSTEM_PROMPT)
                agent_input_messages = [system_message] + conversation_messages # Prepend system prompt

                agent_input = {
                    "messages": agent_input_messages
                }
                
                # Run the react agent subgraph with our input
                agent_response = await react_agent.ainvoke(agent_input)
                
                # Update the state with the new messages (excluding the system prompt we added)
                # The react agent output 'messages' includes the history plus the final answer.
                # We only want to append the *new* messages to our state's history.
                # A robust way is to compare with agent_input_messages
                new_messages_from_agent = []
                if "messages" in agent_response:
                    # Check how many messages were in the input
                    input_len = len(agent_input_messages)
                    # Assume new messages are those beyond the input length in the response
                    # This assumes the react agent returns the full history + new response
                    if len(agent_response["messages"]) > input_len:
                         new_messages_from_agent = agent_response["messages"][input_len:]
                    elif agent_response["messages"] and agent_response["messages"][-1].type == 'ai':
                        # Fallback: assume the last message is the new AI response if lengths don't match as expected
                        new_messages_from_agent = [agent_response["messages"][-1]]

                # Append only the new messages generated by the agent run to the original full history
                # Ensure state["messages"] is treated as a list
                current_history_for_update = state.get("messages", [])
                updated_messages = current_history_for_update + new_messages_from_agent
                
                # 记录性能信息
                response_time = (time.time() - start_time) * 1000
                mcp_logger.log_performance(
                    server_name="agent",
                    operation="chat_node",
                    duration=response_time,
                    success=True
                )
                
                # End the graph with the updated messages
                return Command(
                    goto=END,
                    update={"messages": updated_messages},
                )
    except Exception as e:
        # 记录错误信息
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