import asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, BaseMessage
from langchain_core.callbacks import StdOutCallbackHandler
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
from typing import TypedDict, Annotated, Sequence
import operator

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    
async def main():
    print("--- MCP Test Client for Google Tasks (LangGraph Streaming) ---")
    
    print("\n[Step 1] Initializing Ollama Chat Model")
    try:
        llm = ChatOllama(
            model="qwen3",
            streaming=True, # Ensures the model itself supports streaming
            callbacks=[StdOutCallbackHandler()] # Prints raw tokens to console
        )
        print("Ollama model initiated successfully.")
        
    except Exception as e:
        print(f"Error initializing Ollama: {str(e)}")
        return 
    
    print("\n[Step 2] Configuring MCP server connection...")
    
    mcp_server_config = {
        "gtasks": {
            "command": "node", 
            "args": ["../Google_Tasks_MCP/gtasks-mcp-main/dist/index.js"]
            # Consider adding "cwd" if the server script relies on a specific working directory:
            # "cwd": "/path/to/your/Google_Tasks_MCP/gtasks-mcp-main/"
        }
    }
    print(f"Server configuration: Connect to 'gtasks' via command 'node' using 'stdio'.")
    
    print("\n[Step 3] Attempting to connect to MCP server and load tools...")

    try:
        async with MultiServerMCPClient(mcp_server_config) as mcp_client: # Renamed client to mcp_client for clarity
            print("MCP client session started.")

            tools = mcp_client.get_tools()

            if not tools:
                print("ERROR: No tools were loaded from the MCP server.")
                print("Please check:")
                print("1. The Google Tasks MCP server IS RUNNING in a separate terminal (`npm run start`).")
                print("2. The 'command' and 'args' paths in `mcp_server_config` are correct.")
                print("3. The server did not crash on startup (check its terminal output).")
                return
            
            tool_names = [tool.name for tool in tools]
            print(f"Successfully loaded {len(tools)} tools: {tool_names}")
            
            model_with_tools = llm.bind_tools(tools)
            
            async def call_model_node(state: AgentState):
                print("\n>> Calling LLM...")
                response_message = await model_with_tools.ainvoke(state['messages'])
                return {"messages": [response_message]}

            tool_node = ToolNode(tools)
            
            print("\n[Step 4] Constructing LangGraph agent...")
            workflow = StateGraph(AgentState)
            workflow.add_node("llm", call_model_node)
            workflow.add_node("tools", tool_node)
            workflow.set_entry_point("llm")
            workflow.add_conditional_edges(
                "llm",
                tools_condition
            )
            workflow.add_edge("tools", "llm")
            
            memory_saver = MemorySaver()
            
            graph = workflow.compile(checkpointer=memory_saver)
            print("LangGraph agent constructed successfully.")
            
            print("\n[Step 5] Running test case with streaming...")
            
            user_message_text = "Show me my current tasks pls"
            print(f"\nUser: \"{user_message_text}\"")
            print("Agent response (streaming via astream_events):")

            config = {"configurable": {"thread_id": "mcp-gtasks-thread-1"}}
            
            full_response_printed = False
            async for event in graph.astream_events(
                {"messages": [HumanMessage(content=user_message_text)]},
                config=config,
                version="v1"
            ):
                kind = event["event"]
                
                if kind == "on_chat_model_stream":
                    chunk_content = event["data"]["chunk"].content
                    if chunk_content:
                        # This will print the streamed tokens.
                        # If StdOutCallbackHandler is active, you might see duplicate printing
                        # or slightly different formatting.
                        print(chunk_content, end="", flush=True)
                        full_response_printed = True 
                elif kind == "on_tool_start":
                    if full_response_printed: print() # Newline if LLM was streaming before tool
                    tool_name = event["name"]
                    tool_input = event["data"].get("input")
                    print(f"\n🤖 Starting tool: `{tool_name}` with input: `{tool_input}` ...", flush=True)
                    full_response_printed = False # Reset for next potential LLM stream
                elif kind == "on_tool_end":
                    tool_name = event["name"]
                    # tool_output = event["data"].get("output") # Can be verbose
                    print(f"\n🤖 Tool `{tool_name}` finished.", flush=True)
                    # No explicit print() here, as the next LLM stream will handle newlines
                    full_response_printed = False # Reset for next potential LLM stream
                elif kind == "on_llm_end" and not full_response_printed :
                    # This handles cases where the LLM response is not streamed (e.g. it's just tool calls)
                    # or if the stream was empty. We want to ensure the final AIMessage content is shown.
                    # However, if StdOutCallbackHandler printed everything, this might be redundant.
                    # Let's check the state for the final message.
                    current_state = await agent_graph.aget_state(config)
                    if current_state and current_state.values:
                        final_messages = current_state.values[0]['messages']
                        if final_messages and isinstance(final_messages[-1], AIMessage) and not final_messages[-1].tool_calls:
                            # If StdOutCallbackHandler didn't print, this will.
                            # This logic can be tricky with multiple callbacks.
                            # For now, we rely on StdOutCallbackHandler for the final AIMessage text if it's not a tool call.
                            # print(f"\nFinal AI Message (from on_llm_end if not streamed): {final_messages[-1].content}")
                            pass # Rely on StdOutCallbackHandler for now for non-tool call final message

            print() # Final newline after streaming finishes

            # Optionally, retrieve and print the final state for inspection
            # final_state = await agent_graph.aget_state(config)
            # if final_state and final_state.values:
            #     print("\n--- Final State Messages ---")
            #     for msg in final_state.values[0]['messages']:
            #         print(f"{msg.type}: {msg.content[:200]}...") # Print snippet

            print("\n--- Test case finished ---")
                      
    except ConnectionRefusedError:
        print("ERROR: Connection to MCP server refused. Is the server running?")
        print("Please ensure the Google Tasks MCP server is running (`npm run start` in its directory).")
    except FileNotFoundError:
        print(f"ERROR: Could not find Node.js executable or server script.")
        print("Please check the 'command' and 'args' paths in `mcp_server_config`.")
        print("The relative path used for 'args' implies this script should be run from a specific directory.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\nClient script finished.")
        
if __name__ == "__main__":
    # Ensure you have the necessary libraries installed:
    # pip install langchain-mcp-adapters langgraph langchain-ollama langchain-core
    print("Starting the asynchronous LangGraph client...")
    asyncio.run(main())
