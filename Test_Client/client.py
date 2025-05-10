import asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
from langchain_core.callbacks import StdOutCallbackHandler


async def main():
    print("--- MCP Test Client for Google Tasks ---")
    
    print("[Step 1] Initializing Ollama Chat Model")
    try:
        llm = ChatOllama(
        model="qwen3",
        streaming=True,
        callbacks=[StdOutCallbackHandler()]
        )
        
        print("Ollama model initiated successfuly. ")
        
    except Exception as e:
        print(f"Error: {str(e)}")
    
    print("[Step 2] Configuring MCP server connection...")
    
    mcp_server_config = {
        "gtasks": {
            "command": "node", 
            "args": ["../Google_Tasks_MCP/gtasks-mcp-main/dist/index.js"]
        }
    }
    
    print(f"Server configuration: Connect to 'gtasks' via command using 'stdio'.")
    
    print("\n[Step 3] Attempting to connect to MCP server and load tools...")

    try:
        async with MultiServerMCPClient(mcp_server_config) as client:
            print("MCP client session started.")

            tools = client.get_tools()

            if not tools:
                print("ERROR: No tools were loaded from the MCP server.")
                print("Please check:")
                print("1. The Google Tasks MCP server IS RUNNING in a separate terminal (`npm run start`).")
                print("2. The 'command' and 'args' paths in `mcp_server_config` are correct.")
                print("3. The server did not crash on startup (check its terminal output).")
                return
            
            tool_names = [tool.name for tool in tools]
            print(f"Successfully loaded {len(tools)} tools: {tool_names}")

            print("\n[Step 4] Creating ReAct agent...")
            try:
                agent_executor = create_react_agent(llm, tools)
                print("ReAct agent created successfully.")
            except Exception as e:
                print(f"ERROR: Could not create ReAct agent. Details: {e}")
                return
            
            print("\n[Step 5] Running test cases...")
            print("\n--- Test Case 1: Listing tasks ---")
            try:
                user_message_list = "Could you add a task for getting the nba project done by the 25th of May"
                print(f"Sending to agent: \"{user_message_list}\"")
                response_list = await agent_executor.ainvoke({"messages": [HumanMessage(content=user_message_list)]})

                if response_list and 'messages' in response_list and response_list['messages']:
                    ai_response_content = response_list['messages'][-1].content
                    print("Agent Response (List):")
                    print(ai_response_content)
                else:
                    print("Agent Response (List) was empty or not in the expected format.")
                    print(f"Full response: {response_list}")

            except Exception as e:
                print(f"ERROR during 'Listing tasks' test case: {e}")
                import traceback
                traceback.print_exc()
                      
    except ConnectionRefusedError:
        print("ERROR: Connection to MCP server refused. Is the server running?")
        print("Please ensure the Google Tasks MCP server is running (`npm run start` in its directory).")
    except FileNotFoundError:
        print(f"ERROR: Could not find Node.js executable or server script.")
        print("Please check the 'command' and 'args' paths in `mcp_server_config`.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\nClient script finished.")
        
if __name__ == "__main__":
    print("Starting the asynchronous client...")
    asyncio.run(main())