"use client";

import { useCoAgent } from "@copilotkit/react-core";
import { useEffect, useState } from "react";
import { useLocalStorage } from "../hooks/useLocalStorage";
import { ExampleConfigs } from "./ExampleConfigs";

type ConnectionType = "stdio" | "sse";

interface StdioConfig {
  command: string;
  args: string[];
  transport: "stdio";
  enabled: boolean;
}

interface SSEConfig {
  url: string;
  transport: "sse";
  enabled: boolean;
}

type ServerConfig = StdioConfig | SSEConfig;

// Define a generic type for our state
interface AgentState {
  mcp_config: Record<string, ServerConfig>;
  system_prompt: string;
}

// Local storage key for saving agent state
const STORAGE_KEY = "mcp-agent-state";

const DEFAULT_SYSTEM_PROMPT = `You are a helpful assistant specializing in Sage Intacct products. Assume all questions are related to Sage Intacct unless the user explicitly states otherwise.\n1. Answer questions based on your knowledge of Sage Intacct.\n2. If you need additional information or capabilities to answer (even within the assumed Sage Intacct context), check if any available tools (registered for this session) can help. Use relevant tools when necessary.\n3. When asked questions about Intacct objects (models):\n   - If you do not have the object schema information in the context, first use the 'listAvailableModels' tool to confirm the object name.\n   - Then use the 'getModelDefinition' tool to get the details about the object. When calling 'getModelDefinition', only use the 'name' field and do NOT add a 'type' field or any other field unless explicitly asked to. For example:\n     {\n       "name": "objects/company-config/employee"\n     }\n     Do NOT include 'type', e.g. this is incorrect:\n     {\n       "name": "objects/general-ledger/journal-entry",\n       "type": "object"\n     }\n   - When querying, use the 'executeQuery' tool only after you have the object details.\n   - Only use fields that are present in the object details returned by 'getModelDefinition'.\n   - Do not guess or use any unknown fields in the query.\n   - Anything used in the 'executeQuery' tool must be defined in the schema or tools.\n4. If, after checking your knowledge and available tools, you still cannot answer the question (or if the user explicitly stated the question is *not* about Sage Intacct and you lack the knowledge/tools), state that you cannot provide an answer.\n5. If the user enters '/skills', list all the tools currently available to you in this session, including their names and descriptions.`;
const SYSTEM_PROMPT_STORAGE_KEY = "system-prompt";

const ExternalLink = () => (
  <svg
    xmlns="http://www.w3.org/2000/svg"
    className="w-3 h-3 ml-1"
    fill="none"
    viewBox="0 0 24 24"
    stroke="currentColor"
  >
    <path
      strokeLinecap="round"
      strokeLinejoin="round"
      strokeWidth={2}
      d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14"
    />
  </svg>
);

export function MCPConfigForm() {
  // Use our localStorage hook for persistent storage
  const [savedConfigs, setSavedConfigs] = useLocalStorage<
    Record<string, ServerConfig>
  >(STORAGE_KEY, {});

  // Initialize agent state with the data from localStorage
  const { state: agentState, setState: setAgentState } = useCoAgent<AgentState>(
    {
      name: "sample_agent",
      initialState: {
        mcp_config: savedConfigs,
        system_prompt: DEFAULT_SYSTEM_PROMPT,
      },
    }
  );

  // Simple getter for configs
  const configs = agentState?.mcp_config || {};

  // Simple setter wrapper for configs
  const setConfigs = (newConfigs: Record<string, ServerConfig>) => {
    setAgentState({ ...agentState, mcp_config: newConfigs });
    setSavedConfigs(newConfigs);
  };

  const [serverName, setServerName] = useState("");
  const [connectionType, setConnectionType] = useState<ConnectionType>("stdio");
  const [command, setCommand] = useState("");
  const [args, setArgs] = useState("");
  const [url, setUrl] = useState("");
  const [isLoading, setIsLoading] = useState(true);
  const [showAddServerForm, setShowAddServerForm] = useState(false);
  const [showExampleConfigs, setShowExampleConfigs] = useState(false);

  // System Prompt state and modal
  const [systemPrompt, setSystemPrompt] = useLocalStorage<string>(SYSTEM_PROMPT_STORAGE_KEY, DEFAULT_SYSTEM_PROMPT);
  const [showPromptModal, setShowPromptModal] = useState(false);
  const [promptDraft, setPromptDraft] = useState(systemPrompt);

  // Calculate server statistics
  const totalServers = Object.keys(configs).length;
  const stdioServers = Object.values(configs).filter(
    (config) => config.transport === "stdio"
  ).length;
  const sseServers = Object.values(configs).filter(
    (config) => config.transport === "sse"
  ).length;

  // Set loading to false when state is loaded
  useEffect(() => {
    if (agentState) {
      setIsLoading(false);
    }
  }, [agentState]);

  // When systemPrompt changes, update agent state
  useEffect(() => {
    setAgentState((prev: any) => ({ ...prev, system_prompt: systemPrompt }));
  }, [systemPrompt]);

  const handleExampleConfig = (exampleConfig: Record<string, ServerConfig>) => {
    // Merge the example with existing configs or replace them based on user preference
    if (Object.keys(configs).length > 0) {
      const shouldReplace = window.confirm(
        "Do you want to replace your current configuration with this example? Click 'OK' to replace, or 'Cancel' to merge."
      );

      if (shouldReplace) {
        setConfigs(exampleConfig);
      } else {
        setConfigs({ ...configs, ...exampleConfig });
      }
    } else {
      setConfigs(exampleConfig);
    }

    // Close the examples panel after selection
    setShowExampleConfigs(false);
  };

  const addConfig = () => {
    if (!serverName) return;

    const newConfig =
      connectionType === "stdio"
        ? {
            command,
            args: args.split(" ").filter((arg) => arg.trim() !== ""),
            transport: "stdio" as const,
            enabled: true,
          }
        : {
            url,
            transport: "sse" as const,
            enabled: true,
          };

    setConfigs({
      ...configs,
      [serverName]: newConfig,
    });

    // Reset form
    setServerName("");
    setCommand("");
    setArgs("");
    setUrl("");
    setShowAddServerForm(false);
  };

  const removeConfig = (name: string) => {
    const newConfigs = { ...configs };
    delete newConfigs[name];
    setConfigs(newConfigs);
  };

  if (isLoading) {
    return <div className="p-4">Loading configuration...</div>;
  }

  return (
    <div className="max-w-5xl mx-auto space-y-6">
      {/* Header */}
      <div className="mb-6">
        <div className="flex flex-col sm:flex-row sm:items-center gap-4 mb-1">
          <div className="flex items-center">
            <svg
              xmlns="http://www.w3.org/2000/svg"
              className="h-6 w-6 mr-2 text-gray-700"
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M5 12h14M5 12a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v4a2 2 0 01-2 2M5 12a2 2 0 00-2 2v4a2 2 0 002 2h14a2 2 0 002-2v-4a2 2 0 00-2-2m-2-4h.01M17 16h.01"
              />
            </svg>
            <h1 className="text-3xl sm:text-5xl font-semibold">
              Intacct Copilot Orchestration
            </h1>
          </div>
        </div>
        <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center mt-4 gap-4">
          <div className="flex flex-col sm:flex-row sm:items-center gap-3">
            <p className="text-sm text-gray-600">
            </p>
          </div>
          <button
            onClick={() => setShowAddServerForm(true)}
            className="w-full sm:w-auto px-3 py-1.5 bg-gray-800 text-white rounded-md text-sm font-medium hover:bg-gray-700 flex items-center gap-1 justify-center"
          >
            <svg
              xmlns="http://www.w3.org/2000/svg"
              className="h-4 w-4"
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M12 4v16m8-8H4"
              />
            </svg>
            Add Skill
          </button>
        </div>
      </div>

      {/* Server Statistics */}
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 mb-8">
        <div className="bg-white border rounded-md p-4">
          <div className="text-sm text-gray-500">Total Skills</div>
          <div className="text-3xl font-bold">{totalServers}</div>
        </div>
        <div className="bg-white border rounded-md p-4">
          <div className="text-sm text-gray-500">Stdio Skills</div>
          <div className="text-3xl font-bold">{stdioServers}</div>
        </div>
        <div className="bg-white border rounded-md p-4">
          <div className="text-sm text-gray-500">SSE Skills</div>
          <div className="text-3xl font-bold">{sseServers}</div>
        </div>
      </div>

      {/* Example Configs Button */}
      <div className="mb-4">
        <button
          onClick={() => setShowExampleConfigs(!showExampleConfigs)}
          className="flex items-center text-sm font-medium text-gray-700 hover:text-gray-900"
        >
          <span>Example Configurations</span>
          <svg
            xmlns="http://www.w3.org/2000/svg"
            className={`w-4 h-4 ml-1 transition-transform ${
              showExampleConfigs ? "rotate-180" : ""
            }`}
            fill="none"
            viewBox="0 0 24 24"
            stroke="currentColor"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M19 9l-7 7-7-7"
            />
          </svg>
        </button>

        {showExampleConfigs && (
          <div className="mt-2">
            <ExampleConfigs onSelectConfig={handleExampleConfig} />
          </div>
        )}
      </div>

      {/* Server List */}
      <div className="bg-white border rounded-md p-6">
        <h2 className="text-lg font-semibold mb-4">Skill List</h2>

        {totalServers === 0 ? (
          <div className="text-gray-500 text-center py-10">
            No skills configured. Click &quot;Add Skill&quot; to get started.
          </div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {Object.entries(configs).map(([name, config]) => (
              <div
                key={name}
                className="border rounded-md overflow-hidden bg-white shadow-sm"
              >
                <div className="p-4">
                  <div className="flex justify-between items-start">
                    <div>
                      <h3 className="font-semibold">{name}</h3>
                      <div className="inline-flex items-center px-2 py-0.5 bg-gray-100 text-xs rounded mt-1">
                        {config.transport === "stdio" ? (
                          <svg
                            xmlns="http://www.w3.org/2000/svg"
                            className="w-3 h-3 mr-1"
                            fill="none"
                            viewBox="0 0 24 24"
                            stroke="currentColor"
                          >
                            <path
                              strokeLinecap="round"
                              strokeLinejoin="round"
                              strokeWidth={2}
                              d="M8 9l3 3-3 3m5 0h3M5 20h14a2 2 0 002-2V6a2 2 0 00-2-2H5a2 2 0 00-2 2v12a2 2 0 002 2z"
                            />
                          </svg>
                        ) : (
                          <svg
                            xmlns="http://www.w3.org/2000/svg"
                            className="w-3 h-3 mr-1"
                            fill="none"
                            viewBox="0 0 24 24"
                            stroke="currentColor"
                          >
                            <path
                              strokeLinecap="round"
                              strokeLinejoin="round"
                              strokeWidth={2}
                              d="M21 12a9 9 0 01-9 9m9-9a9 9 0 00-9-9m9 9H3m9 9a9 9 0 01-9-9m9 9c1.657 0 3-4.03 3-9s-1.343-9-3-9m0 18c-1.657 0-3-4.03-3-9s1.343-9 3-9m-9 9a9 9 0 019-9"
                            />
                          </svg>
                        )}
                        {config.transport}
                      </div>
                    </div>
                    <div className="flex items-center gap-2">
                      <button
                        onClick={() => {
                          const updatedConfig = {
                            ...config,
                            enabled: !config.enabled
                          };
                          setConfigs({
                            ...configs,
                            [name]: updatedConfig
                          });
                        }}
                        className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors focus:outline-none focus:ring-2 focus:ring-gray-500 focus:ring-offset-2 ${
                          config.enabled ? 'bg-gray-800' : 'bg-gray-200'
                        }`}
                      >
                        <span
                          className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${
                            config.enabled ? 'translate-x-6' : 'translate-x-1'
                          }`}
                        />
                      </button>
                      <button
                        onClick={() => removeConfig(name)}
                        className="text-gray-400 hover:text-red-500"
                      >
                        <svg
                          xmlns="http://www.w3.org/2000/svg"
                          className="w-4 h-4"
                          fill="none"
                          viewBox="0 0 24 24"
                          stroke="currentColor"
                        >
                          <path
                            strokeLinecap="round"
                            strokeLinejoin="round"
                            strokeWidth={2}
                            d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16"
                          />
                        </svg>
                      </button>
                    </div>
                  </div>
                  <div className="mt-3 text-sm text-gray-600">
                    {config.transport === "stdio" ? (
                      <>
                        <p>Command: {config.command}</p>
                        <p className="truncate">
                          Args: {config.args.join(" ")}
                        </p>
                      </>
                    ) : (
                      <p className="truncate">URL: {config.url}</p>
                    )}
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Import/Export Buttons */}
      <div className="flex gap-2 justify-end mt-4">
        <button
          onClick={() => { setPromptDraft(systemPrompt); setShowPromptModal(true); }}
          className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 text-sm font-medium flex items-center"
        >
          <svg
            xmlns="http://www.w3.org/2000/svg"
            className="w-4 h-4 mr-1"
            fill="none"
            viewBox="0 0 24 24"
            stroke="currentColor"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M12 4v16m8-8H4"
            />
          </svg>
          System Prompt
        </button>
        <button
          onClick={async () => {
            try {
              await navigator.clipboard.writeText(
                JSON.stringify(configs, null, 2)
              );
              alert("Exported MCP Servers to clipboard as JSON.");
            } catch (err) {
              alert("Failed to export to clipboard: " + err);
            }
          }}
          className="px-4 py-2 bg-gray-800 text-white rounded-md hover:bg-gray-700 text-sm font-medium flex items-center"
        >
          <svg
            xmlns="http://www.w3.org/2000/svg"
            className="w-4 h-4 mr-1"
            fill="none"
            viewBox="0 0 24 24"
            stroke="currentColor"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M8 17l4 4 4-4m-4-5v9"
            />
          </svg>
          Export
        </button>
        <button
          onClick={async () => {
            try {
              const text = await navigator.clipboard.readText();
              const json = JSON.parse(text);
              if (typeof json === "object" && json !== null) {
                setConfigs(json);
                alert("Imported MCP Servers from clipboard.");
              } else {
                alert("Clipboard does not contain a valid MCP Servers JSON config.");
              }
            } catch (err) {
              alert("Failed to import from clipboard: " + err);
            }
          }}
          className="px-4 py-2 bg-gray-800 text-white rounded-md hover:bg-gray-700 text-sm font-medium flex items-center"
        >
          <svg
            xmlns="http://www.w3.org/2000/svg"
            className="w-4 h-4 mr-1"
            fill="none"
            viewBox="0 0 24 24"
            stroke="currentColor"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M12 4v16m8-8H4"
            />
          </svg>
          Import
        </button>
      </div>

      {/* Add Server Modal */}
      {showAddServerForm && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-white rounded-lg p-6 w-full max-w-md">
            <div className="flex justify-between items-center mb-4">
              <h2 className="text-lg font-semibold flex items-center">
                <svg
                  xmlns="http://www.w3.org/2000/svg"
                  className="w-5 h-5 mr-2"
                  fill="none"
                  viewBox="0 0 24 24"
                  stroke="currentColor"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M12 4v16m8-8H4"
                  />
                </svg>
                Add New Skill
              </h2>
              <button
                onClick={() => setShowAddServerForm(false)}
                className="text-gray-500 hover:text-gray-700"
              >
                <svg
                  xmlns="http://www.w3.org/2000/svg"
                  className="w-5 h-5"
                  fill="none"
                  viewBox="0 0 24 24"
                  stroke="currentColor"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M6 18L18 6M6 6l12 12"
                  />
                </svg>
              </button>
            </div>

            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium mb-1">
                  Skill Name
                </label>
                <input
                  type="text"
                  value={serverName}
                  onChange={(e) => setServerName(e.target.value)}
                  className="w-full px-3 py-2 border rounded-md text-sm"
                  placeholder="e.g., api-service, data-processor"
                />
              </div>

              <div>
                <label className="block text-sm font-medium mb-1">
                  Connection Type
                </label>
                <div className="grid grid-cols-2 gap-2">
                  <button
                    type="button"
                    onClick={() => setConnectionType("stdio")}
                    className={`px-3 py-2 border rounded-md text-center flex items-center justify-center ${
                      connectionType === "stdio"
                        ? "bg-gray-200 border-gray-400 text-gray-800"
                        : "bg-white text-gray-700"
                    }`}
                  >
                    <svg
                      xmlns="http://www.w3.org/2000/svg"
                      className="w-4 h-4 mr-1"
                      fill="none"
                      viewBox="0 0 24 24"
                      stroke="currentColor"
                    >
                      <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        strokeWidth={2}
                        d="M8 9l3 3-3 3m5 0h3M5 20h14a2 2 0 002-2V6a2 2 0 00-2-2H5a2 2 0 00-2 2v12a2 2 0 002 2z"
                      />
                    </svg>
                    Standard IO
                  </button>
                  <button
                    type="button"
                    onClick={() => setConnectionType("sse")}
                    className={`px-3 py-2 border rounded-md text-center flex items-center justify-center ${
                      connectionType === "sse"
                        ? "bg-gray-200 border-gray-400 text-gray-800"
                        : "bg-white text-gray-700"
                    }`}
                  >
                    <svg
                      xmlns="http://www.w3.org/2000/svg"
                      className="w-4 h-4 mr-1"
                      fill="none"
                      viewBox="0 0 24 24"
                      stroke="currentColor"
                    >
                      <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        strokeWidth={2}
                        d="M21 12a9 9 0 01-9 9m9-9a9 9 0 00-9-9m9 9H3m9 9a9 9 0 01-9-9m9 9c1.657 0 3-4.03 3-9s-1.343-9-3-9m0 18c-1.657 0-3-4.03-3-9s1.343-9 3-9m-9 9a9 9 0 019-9"
                      />
                    </svg>
                    SSE
                  </button>
                </div>
              </div>

              {connectionType === "stdio" ? (
                <>
                  <div>
                    <label className="block text-sm font-medium mb-1">
                      Command
                    </label>
                    <input
                      type="text"
                      value={command}
                      onChange={(e) => setCommand(e.target.value)}
                      className="w-full px-3 py-2 border rounded-md text-sm"
                      placeholder="e.g., python, node"
                    />
                  </div>
                  <div>
                    <label className="block text-sm font-medium mb-1">
                      Arguments
                    </label>
                    <input
                      type="text"
                      value={args}
                      onChange={(e) => setArgs(e.target.value)}
                      className="w-full px-3 py-2 border rounded-md text-sm"
                      placeholder="e.g., path/to/script.py"
                    />
                  </div>
                </>
              ) : (
                <div>
                  <label className="block text-sm font-medium mb-1">URL</label>
                  <input
                    type="text"
                    value={url}
                    onChange={(e) => setUrl(e.target.value)}
                    className="w-full px-3 py-2 border rounded-md text-sm"
                    placeholder="e.g., http://localhost:8000/events"
                  />
                </div>
              )}

              <div className="flex justify-end space-x-2 pt-2">
                <button
                  onClick={() => setShowAddServerForm(false)}
                  className="px-4 py-2 border text-gray-700 rounded-md hover:bg-gray-50 text-sm font-medium flex items-center"
                >
                  <svg
                    xmlns="http://www.w3.org/2000/svg"
                    className="w-4 h-4 mr-1"
                    fill="none"
                    viewBox="0 0 24 24"
                    stroke="currentColor"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M6 18L18 6M6 6l12 12"
                    />
                  </svg>
                  Cancel
                </button>
                <button
                  onClick={addConfig}
                  className="px-4 py-2 bg-gray-800 text-white rounded-md hover:bg-gray-700 text-sm font-medium flex items-center"
                >
                  <svg
                    xmlns="http://www.w3.org/2000/svg"
                    className="w-4 h-4 mr-1"
                    fill="none"
                    viewBox="0 0 24 24"
                    stroke="currentColor"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M12 4v16m8-8H4"
                    />
                  </svg>
                  Add Skill
                </button>
              </div>
            </div>
          </div>
        </div>
      )}

      {showPromptModal && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black bg-opacity-40">
          <div className="bg-white rounded-lg shadow-lg p-6 w-full max-w-2xl relative">
            <h2 className="text-lg font-semibold mb-2">Edit System Prompt</h2>
            <textarea
              className="w-full h-48 border rounded-md p-2 text-sm mb-4"
              value={promptDraft}
              onChange={e => setPromptDraft(e.target.value)}
            />
            <div className="flex justify-end gap-2">
              <button
                onClick={() => setShowPromptModal(false)}
                className="px-4 py-2 border text-gray-700 rounded-md hover:bg-gray-50 text-sm font-medium"
              >
                Cancel
              </button>
              <button
                onClick={() => { setSystemPrompt(promptDraft); setShowPromptModal(false); }}
                className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 text-sm font-medium"
              >
                Save
              </button>
            </div>
            <button
              onClick={() => setShowPromptModal(false)}
              className="absolute top-2 right-2 text-gray-400 hover:text-gray-600"
            >
              <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
