import asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_ollama import ChatOllama 
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, BaseMessage 
from langchain_core.callbacks import StdOutCallbackHandler
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode 
from langgraph.checkpoint.memory import MemorySaver
from typing import TypedDict, Annotated, Sequence 
import operator 

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]

async def main():
    print("--- MCP Test Client for Google Tasks (LangGraph Streaming & Conversational) ---")
    
    print("\n[Step 1] Initializing Ollama Chat Model")
    try:
        llm = ChatOllama(
            model="qwen3",
            streaming=True, 
            callbacks=[StdOutCallbackHandler()] 
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
        }
    }
    print(f"Server configuration: Connect to 'gtasks' via command 'node' using 'stdio'.")
    
    print("\n[Step 3] Attempting to connect to MCP server and load tools...")

    try:
        async with MultiServerMCPClient(mcp_server_config) as mcp_client:
            print("MCP client session started.")

            tools = mcp_client.get_tools()

            if not tools:
                print("ERROR: No tools were loaded from the MCP server.")
                return
            
            tool_names = [tool.name for tool in tools]
            print(f"Successfully loaded {len(tools)} tools: {tool_names}")

            model_with_tools = llm.bind_tools(tools)

            async def call_model_node(state: AgentState):
                print("\n>> Calling LLM...")
                response_message = await model_with_tools.ainvoke(state['messages'])
                return {"messages": [response_message]}

            tool_node = ToolNode(tools)

            def should_continue(state: AgentState):
                last_message = state['messages'][-1]
                if hasattr(last_message, "tool_calls") and last_message.tool_calls:
                    print(">> LLM decided to use a tool.")
                    return "tools"
                else:
                    print(">> LLM provided a direct answer or finished tool use.")
                    return END

            print("\n[Step 4] Constructing LangGraph agent...")
            workflow = StateGraph(AgentState)
            workflow.add_node("llm", call_model_node)
            workflow.add_node("tools", tool_node)
            workflow.set_entry_point("llm")
            workflow.add_conditional_edges(
                "llm",
                should_continue,
                {"tools": "tools", END: END}
            )
            workflow.add_edge("tools", "llm")
            
            memory_saver = MemorySaver() 
            agent_graph = workflow.compile(checkpointer=memory_saver)
            print("LangGraph agent constructed successfully.")
            
            print("\n[Step 5] Starting conversational loop (type 'exit' or 'quit' to end)...")
            
            conversation_thread_id = "mcp-gtasks-conversation-1"
            config = {"configurable": {"thread_id": conversation_thread_id}}

            while True:
                user_input = input("\nYou: ")
                if user_input.lower() in ["exit", "quit"]:
                    print("Exiting conversation.")
                    break
                if not user_input.strip():
                    continue

                print("Agent (streaming via astream_events): ", end="")
                
                full_response_printed_this_turn = False
                async for event in agent_graph.astream_events(
                    {"messages": [HumanMessage(content=user_input)]}, 
                    config=config, 
                    version="v1" 
                ):
                    kind = event["event"]
                    
                    if kind == "on_chat_model_stream":
                        chunk_content = event["data"]["chunk"].content
                        if chunk_content:
                            print(chunk_content, end="", flush=True)
                            full_response_printed_this_turn = True
                    elif kind == "on_tool_start":
                        if full_response_printed_this_turn: print() 
                        tool_name = event["name"]
                        tool_input = event["data"].get("input")
                        print(f"\n🤖 Starting tool: `{tool_name}` with input: `{tool_input}` ...", flush=True)
                        full_response_printed_this_turn = False 
                    elif kind == "on_tool_end":
                        tool_name = event["name"]
                        print(f"\n🤖 Tool `{tool_name}` finished.", flush=True)
                        full_response_printed_this_turn = False
                print() 

            print("\n--- Conversation finished ---")
                      
    except ConnectionRefusedError:
        print("ERROR: Connection to MCP server refused. Is the server running?")
    except FileNotFoundError:
        print(f"ERROR: Could not find Node.js executable or server script.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\nClient script finished.")
        
if __name__ == "__main__":
    print("Starting the asynchronous LangGraph client...")
    asyncio.run(main())