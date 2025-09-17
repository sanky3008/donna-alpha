#!/usr/bin/env python3

# Import all necessary libraries
from langgraph.prebuilt import create_react_agent
from langgraph_supervisor import create_supervisor
from IPython.display import Image, display
from langchain_core.runnables.config import RunnableConfig
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_ollama import ChatOllama
from langgraph.checkpoint.sqlite import SqliteSaver
from dotenv import load_dotenv
from langchain_core.tools import tool
import uuid
import sys
import os
from sqlite3 import connect

# Load environment variables
load_dotenv()

# Global variables to store state
current_user_id = None
current_thread_id = None
vectorstore = None
notes_llm = None
supervisor_llm = None
notes_tools = None
notes_agent_graph = None
supervisor_agent_graph = None

def initialize_models():
    """Initialize the language models and embeddings"""
    global notes_llm, supervisor_llm, vectorstore
    
    # Initialize models
    notes_llm = ChatOpenAI(model="gpt-5-nano", temperature=0)
    supervisor_llm = ChatOpenAI(model="gpt-5-nano", temperature=0)
    
    # Initialize embeddings and vectorstore
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = Chroma(
        collection_name="notes",
        embedding_function=embeddings,
        persist_directory="./data/chroma_db"
    )

# Define tools at module level to ensure proper scope
@tool
def create_note(note: str):
    """
    Creates a new note in memory

    Args:
        note: str
    """
    global current_user_id, vectorstore
    # Use stored user context
    metadata = {"user_id": current_user_id} if current_user_id else {}
    vectorstore.add_texts([note], metadatas=[metadata])
    return f"Note added: {note}"

@tool
def read_note(query: str, k: int = 3):
    """
    Finds top k notes similar to a given query

    Args:
        query: str
        k: number of notes to return (default: 3)
    """
    global current_user_id, vectorstore
    # Use stored user context for filtering
    filter_dict = {}
    if current_user_id:
        filter_dict = {"user_id": current_user_id}

    if filter_dict:
        results = vectorstore.similarity_search(query, k=k, filter=filter_dict)
    else:
        results = vectorstore.similarity_search(query, k=k)

    return results

@tool
def get_all_notes():
    """
    Retrieves all notes for a user
    """
    global current_user_id, vectorstore
    # Use stored user context for filtering
    filter_dict = {}
    if current_user_id:
        filter_dict = {"user_id": current_user_id}

    # Get all documents by using a very generic query
    if filter_dict:
        results = vectorstore.similarity_search("", k=1000, filter=filter_dict)
    else:
        results = vectorstore.similarity_search("", k=1000)

    return results

@tool
def update_note(note_id: str, new_text: str):
    """
    Updates an existing note

    Args:
        note_id: str
        new_text: str
    """
    global current_user_id, vectorstore
    # Use stored user context
    metadata = {"user_id": current_user_id} if current_user_id else {}

    vectorstore.add_texts(
        texts=[new_text],
        ids=[note_id],
        metadatas=[metadata]
    )

    return f"Note {note_id} updated. New note: {new_text}"

@tool
def delete_note(note_id: str):
    """
    Deletes a note

    Args:
        note_id: str
    """
    global vectorstore
    vectorstore.delete(ids=[note_id])
    return f"Note {note_id} deleted"

def create_notes_tools():
    """Return the list of notes management tools"""
    return [create_note, read_note, update_note, delete_note, get_all_notes]

def build_notes_agent():
    """Build the notes agent graph"""
    global notes_llm, notes_tools, notes_agent_graph
    
    # System message for the notes agent
    notes_sys_msg = """You are the Notes Agent, a specialized tool-using agent for managing user notes in a vector database.

CRITICAL INSTRUCTIONS:
- You MUST use the available tools to perform any note-related operations
- NEVER respond with plain text - ALWAYS use tool calls
- You have these tools available: create_note, read_note, get_all_notes, update_note, delete_note
- When asked to list or retrieve notes, use the get_all_notes tool
- When asked to search for specific notes, use the read_note tool
- When asked to save a note, use the create_note tool

EXAMPLE BEHAVIOR:
User request: "list all my notes"
Your response: Call get_all_notes tool immediately

You are interacting with Donna (supervisor agent) who will relay your tool results to the user."""
    
    # Create the prebuilt React agent without checkpointer (will be handled by supervisor)
    notes_agent_graph = create_react_agent(
        model=notes_llm,
        tools=notes_tools,
        prompt=notes_sys_msg,
        name ="notes-agent"
    )

def build_supervisor_agent():
    """Build the supervisor agent graph"""
    global supervisor_llm, notes_agent_graph, supervisor_agent_graph
    
    # Ensure data directory exists
    os.makedirs("./data", exist_ok=True)
    
    # Initialize SQLite checkpointer with direct connection
    db_path = "./data/checkpoints.db"
    conn = connect(db_path, check_same_thread=False)
    checkpointer = SqliteSaver(conn)

    # System message for Donna (the supervisor)
    supervisor_sys_msg = """You are Donna, a helpful, reliable supervisor agent. 
- You directly interact with the user. 
- If the user asks about notes (creating, searching, updating, deleting), you should delegate to the notes_agent.
- For other general queries, you can respond normally like a general AI assistant.
- Be proactive in clarifying user intent if their request is ambiguous.
- Do not expose internal reasoning or tool details to the user, just keep responses natural.
"""

    # Define the subagents for the supervisor
    subagents = [notes_agent_graph]

    # Create the prebuilt supervisor agent
    graph = create_supervisor(
        model=supervisor_llm,
        agents=subagents,
        prompt=supervisor_sys_msg,
    )

    supervisor_agent_graph = graph.compile(checkpointer=checkpointer)

def initialize_agent_system():
    """Initialize the complete agent system"""
    global notes_tools
    
    # Initialize models and vectorstore
    initialize_models()
    
    # Create notes tools
    notes_tools = create_notes_tools()
    
    # Build agents
    build_notes_agent()
    build_supervisor_agent()

def invoke_supervisor(messages, config: RunnableConfig = None):
    """
    Invoke the supervisor agent with messages
    
    Args:
        messages: List of messages or single message
        config: RunnableConfig with user_id in configurable
    """
    global current_user_id, current_thread_id, supervisor_agent_graph
    
    if not isinstance(messages, list):
        messages = [messages]
    
    # Extract and store user context for tools to use
    if config:
        current_user_id = config["configurable"]['user_id']
        current_thread_id = config["configurable"]['thread_id']
    
    return supervisor_agent_graph.invoke(
        {"messages": messages},
        config=config
    )

def display_supervisor_graph():
    """Displays the supervisor graph visualization"""
    global supervisor_agent_graph
    display(Image(supervisor_agent_graph.get_graph(xray=True).draw_mermaid_png()))

def display_notes_graph():
    """Displays the notes agent graph visualization"""
    global notes_agent_graph
    display(Image(notes_agent_graph.get_graph(xray=True).draw_mermaid_png()))

def main():
    """Main chat function"""
    print("🤖 Supervisor Agent Chat")
    print("Type 'quit' or 'exit' to end the conversation")
    print("=" * 50)
    
    # Ensure data directory exists
    os.makedirs("./data", exist_ok=True)
    
    # Initialize the agent system
    initialize_agent_system()
    
    # Generate a unique user ID and thread ID for this session
    user_id = "009"
    thread_id = "terminal"
    
    # Create config with user_id and thread_id for checkpointer
    # Use checkpoint_ns to isolate users' checkpoint histories
    config = {
        'configurable': {
            "thread_id": f"{user_id}_{thread_id}",
            "user_id": user_id # Use user_id as checkpoint namespace for isolation
        }
    }
    
    # Chat loop
    while True:
        try:
            # Get user input
            user_input = input("\n👤 You: ").strip()

            # Check for exit commands
            if user_input.lower() in ['quit', 'exit', 'bye']:
                print("\n👋 Goodbye!")
                break

            if not user_input:
                continue

            # Create message and invoke supervisor
            message = HumanMessage(content=user_input)
            result = invoke_supervisor([message], config=config)

            # Display agent response
            response = result['messages'][-1].content
            print(f"\n🤖 Assistant: {response}")

        except KeyboardInterrupt:
            print("\n\n👋 Chat interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            print("Please try again or type 'quit' to exit.")

if __name__ == "__main__":
    main()
