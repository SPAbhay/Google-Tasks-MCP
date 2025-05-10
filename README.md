# Google-Tasks-MCP

This project demonstrates a conversational agent capable of interacting with Google Tasks. It comprises two main components:

1.  **Google Tasks MCP Server (`Google_Tasks_MCP/gtasks-mcp-main/`)**: A Node.js server that implements the Model Context Protocol (MCP) to expose Google Tasks functionalities as tools.
2.  **Conversational Test Client (`Test_Client/`)**: A Python-based client using LangChain, LangGraph, and a local Ollama model (`qwen3`) to interact with the MCP server, allowing users to manage their Google Tasks through natural language.

## Prerequisites

* **Node.js and npm**: For running the Google Tasks MCP Server.
* **Python 3.8+**: For running the Test Client.
* **Ollama**: With the `qwen3` model pulled (`ollama pull qwen3`). Ensure Ollama is running (`ollama serve`).
* **Google Cloud Project**: With the Google Tasks API enabled and OAuth 2.0 credentials configured (see server setup).

## Setup

### 1. Google Tasks MCP Server (`Google_Tasks_MCP/gtasks-mcp-main/`)

The setup for this server is based on the instructions typically provided for such MCP servers.

1.  **Google Cloud Project Configuration**:
    * Create a new Google Cloud project.
    * Enable the **Google Tasks API**.
    * Configure an **OAuth consent screen**. For testing, "internal" user type might be sufficient if using a GSuite account, otherwise "external" and add test users.
    * Add the scope: `https://www.googleapis.com/auth/tasks`.
    * Create an **OAuth 2.0 Client ID** for application type "Desktop App".
    * Download the JSON file of your client's OAuth keys.

2.  **Server Files and Authentication**:
    * Navigate to the server directory: `cd Google_Tasks_MCP/gtasks-mcp-main`
    * Rename the downloaded OAuth key file to `gcp-oauth.keys.json` and place it in this `Google_Tasks_MCP/gtasks-mcp-main/` directory.
    * Install dependencies: `npm install`
    * Build the server: `npm run build` (this should create a `dist` folder).
    * Authenticate the server with Google: `npm run start auth`
        * This will open an authentication flow in your system browser. Complete the process.
        * Credentials (an access token, etc.) should be saved in a file like `.gtasks-server-credentials.json` in the server's root directory (this file is gitignored).

### 2. Conversational Test Client (`Test_Client/`)

1.  **Navigate to the client directory**:
    ```bash
    cd Test_Client
    ```
2.  **Create and activate a Python virtual environment (recommended)**:
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows: .venv\Scripts\activate
    ```
3.  **Install Python dependencies**:
    ```bash
    pip install langchain-mcp-adapters langgraph langchain-ollama langchain-core
    ```
4.  **Configure Client Script**:
    * Open `Test_Client/client.py`.
    * Ensure the `mcp_server_config` within `client.py` correctly points to your Node.js executable and the `dist/index.js` script of the Google Tasks MCP server. The path to `dist/index.js` should be absolute or correctly relative to where you run the Python script.
        ```python
        mcp_server_config = {
            "gtasks": {
                "command": "node", # Or your specific path to node, e.g., "/opt/homebrew/bin/node"
                "args": ["../Google_Tasks_MCP/gtasks-mcp-main/dist/index.js"] # Adjust this path if necessary
            }
        }
        ```
        *(The provided relative path assumes `Test_Client` and `Google_Tasks_MCP` are sibling directories under the main `MCP` project root, and you run `client.py` from within `Test_Client`)*.

## Running the Project

1.  **Start the Ollama Server**:
    * Open a new terminal and run:
        ```bash
        ollama serve
        ```
    * Ensure the `qwen3` model is available (`ollama list`).

2.  **Start the Google Tasks MCP Server**:
    * Open another new terminal.
    * Navigate to the server directory: `cd Google_Tasks_MCP/gtasks-mcp-main`
    * Run:
        ```bash
        npm run start
        ```
    * This server will remain running, listening for connections from the client.

3.  **Run the Python Conversational Client**:
    * Open a third new terminal.
    * Navigate to the client directory: `cd Test_Client`
    * Activate your Python virtual environment (if you created one): `source .venv/bin/activate`
    * Run the client script:
        ```bash
        python client.py
        ```
    * You should now be able to interact with the agent by typing messages in the terminal. Type "exit" or "quit" to end the conversation.

## Key Technologies

* **Model Context Protocol (MCP)**: For standardizing tool use by language models.
* **Node.js / TypeScript**: For the Google Tasks MCP Server.
* **Python**: For the conversational client.
* **LangChain / LangGraph**: Frameworks for building LLM applications.
* **Ollama (`qwen3` model)**: For local LLM inference.
* **Google Tasks API**: For interacting with Google Tasks.
