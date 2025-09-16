from langgraph.graph import START, StateGraph, MessagesState
from langgraph.prebuilt import tools_condition
from langgraph.prebuilt import ToolNode
from IPython.display import Image, display
from langgraph.graph import MessagesState
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.runnables.config import RunnableConfig
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_ollama import ChatOllama
from langgraph.checkpoint.sqlite import SqliteSaver


from dotenv import load_dotenv

load_dotenv()

class NotesAgent:
    def __init__(self):
        self.llm = ChatOllama(model="llama3.2:latest", temperature=0)
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        
        # Store current user context (will be set when invoke is called)
        self.current_user_id = None
        
        # Create vectorstore
        self.vectorstore = Chroma(
            collection_name="notes",
            embedding_function=self.embeddings,
            persist_directory="./data/chroma_db"
        )
        
        # Initialize tools and graph
        self.tools = [self.create_note, self.read_note, self.update_note, self.delete_note, self.get_all_notes]
        self.llm_with_tools = self.llm.bind_tools(self.tools)
        
        self.sys_msg = SystemMessage(
            content=
            """You are the Notes Agent, responsible for managing user notes in a vector database. 
- You only interact through tool calls, not directly with the user. 
- You interact with Donna, who is the assistant facing the user. You should do whatever she says. 
- You have five tools: Create Note, Retrieve Notes, Update Note, Delete Note and Get All Notes.
- Always respond with the minimal necessary tool call when invoked. 
- Do not generate any user-facing text, just perform the requested operation and return results for Donna to relay.
"""
        )
        # Build the graph
        self._build_graph()

    def create_note(self, note: str):
        """
        Creates a new note in memory

        Args:
            note: str
        """
        # Use stored user context
        metadata = {"user_id": self.current_user_id} if self.current_user_id else {}

        self.vectorstore.add_texts([note], metadatas=[metadata])

        return f"Note added: {note}"

    def read_note(self, query: str, k: int = 3):
        """
        Finds top k notes similar to a given query

        Args:
            query: str
            k: number of notes to return (default: 3)
        """
        # Use stored user context for filtering
        filter_dict = {}
        if self.current_user_id:
            filter_dict = {"user_id": self.current_user_id}

        if filter_dict:
            results = self.vectorstore.similarity_search(query, k=k, filter=filter_dict)
        else:
            results = self.vectorstore.similarity_search(query, k=k)

        return results

    def get_all_notes(self):
        """
        Retrieves all notes for a user
        """
        # Use stored user context for filtering
        filter_dict = {}
        if self.current_user_id:
            filter_dict = {"user_id": self.current_user_id}

        # Get all documents by using a very generic query
        if filter_dict:
            results = self.vectorstore.similarity_search("", k=1000, filter=filter_dict)
        else:
            results = self.vectorstore.similarity_search("", k=1000)

        return results

    def update_note(self, note_id: str, new_text: str):
        """
        Updates an existing note

        Args:
            note_id: str
            new_text: str
        """
        # Use stored user context
        metadata = {"user_id": self.current_user_id} if self.current_user_id else {}

        self.vectorstore.add_texts(
            texts=[new_text],
            ids=[note_id],
            metadatas=[metadata]
        )

        return f"Note {note_id} updated. New note: {new_text}"

    def delete_note(self, note_id: str):
        """
        Deletes a note

        Args:
            note_id: str
        """
        self.vectorstore.delete(ids=[note_id])

        return f"Note {note_id} deleted"

    def assistant(self, state: MessagesState, config: RunnableConfig):
        return {"messages": [self.llm_with_tools.invoke([self.sys_msg] + state["messages"], config=config)]}

    def _build_graph(self):
        builder = StateGraph(MessagesState)

        builder.add_node("assistant", self.assistant)
        builder.add_node("tools", ToolNode(self.tools))

        builder.add_edge(START, "assistant")
        builder.add_conditional_edges("assistant", tools_condition)
        builder.add_edge("tools", "assistant")
        # Use SQLite checkpointer in the same data directory as vector db
        import os
        from sqlite3 import connect
        os.makedirs("./data", exist_ok=True)
        
        # Initialize SQLite checkpointer with direct connection
        db_path = "./data/notes_checkpoints.db"
        conn = connect(db_path, check_same_thread=False)
        checkpointer = SqliteSaver(conn)

        self.agent_graph = builder.compile(checkpointer=checkpointer)

    def get_graph(self):
        """Returns the compiled agent graph"""
        return self.agent_graph

    def display_graph(self):
        """Displays the graph visualization"""
        display(Image(self.agent_graph.get_graph(xray=True).draw_mermaid_png()))

    def invoke(self, messages, config: RunnableConfig):
        """
        Invoke the notes agent with messages
        
        Args:
            messages: List of messages or single message
            config: RunnableConfig with user_id in configurable
        """
        if not isinstance(messages, list):
            messages = [messages]
        
        # Extract and store user context for tools to use
        if config:
            self.current_user_id = config["configurable"]["user_id"]
        
        return self.agent_graph.invoke(
            {"messages": messages},
            config=config
        )

# Create a default instance for backward compatibility
def create_notes_agent():
    """Factory function to create a new NotesAgent instance"""
    return NotesAgent()

# For backward compatibility and direct usage
if __name__ == "__main__":
    notes_agent = NotesAgent()
    notes_agent.display_graph()