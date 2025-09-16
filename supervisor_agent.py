from langgraph.graph import START, StateGraph, MessagesState
from langgraph.prebuilt import tools_condition
from langgraph.prebuilt import ToolNode
from IPython.display import Image, display
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_core.runnables.config import RunnableConfig
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from notes_agent import NotesAgent
from dotenv import load_dotenv

load_dotenv()

class SupervisorAgent:
    def __init__(self):
        # self.llm = ChatOllama(model="llama3.2:latest", temperature=0)
        self.llm = ChatOpenAI(model="gpt-5-nano", temperature=0)
        
        # Initialize notes agent
        self.notes_agent = NotesAgent()
        
        # Store current user context (will be set when invoke is called)
        self.current_user_id = None
        self.current_thread_id = None
        self.current_state = None  # Store current MessagesState for tools to access
        
        # Initialize tools and graph
        self.tools = [self.delegate_to_notes]
        self.llm_with_tools = self.llm.bind_tools(self.tools)
        
        self.sys_msg = SystemMessage(
            content="""You are Donna, a helpful, reliable supervisor agent. 
- You directly interact with the user. 
- If the user asks about notes (creating, searching, updating, deleting), you should NOT try to do it yourself. Instead, you should call the Notes Agent using the appropriate tool.
- For other general queries, you can respond normally like a general AI assistant.
- Be proactive in clarifying user intent if their request is ambiguous.
- Do not expose internal reasoning or tool details to the user, just keep responses natural.
""")
        # Build the graph
        self._build_graph()

    def delegate_to_notes(self, task: str):
        """
        Delegates note-related tasks to the notes agent

        Args:
            task: str - The task to delegate to notes agent
        """
        # Get the full message history from current state
        full_messages = self.current_state["messages"] if self.current_state else []
        
        # Add the specific task as a new message to the conversation
        task_message = HumanMessage(content=f"Donna: {task}")
        messages_with_task = full_messages + [task_message]
        
        # Create config object with user isolation using stored context
        config = {
            "configurable": {
                "user_id": self.current_user_id,
                "thread_id": f"notes_{self.current_thread_id}",  # Prefix to distinguish from supervisor thread
                "checkpoint_ns": self.current_user_id
            }
        }
        
        # Delegate to notes agent with full conversation history + specific task
        result = self.notes_agent.invoke(messages_with_task, config=config)
        
        return f"Notes agent result: {result['messages'][-1].content}"

    def assistant(self, state: MessagesState, config: RunnableConfig):
        # Store current state for tools to access
        self.current_state = state
        
        return {"messages": [self.llm_with_tools.invoke([self.sys_msg] + state["messages"], config=config)]}

    def _build_graph(self):
        builder = StateGraph(MessagesState)

        builder.add_node("assistant", self.assistant)
        builder.add_node("tools", ToolNode(self.tools))

        builder.add_edge(START, "assistant")
        builder.add_conditional_edges("assistant", tools_condition)
        builder.add_edge("tools", "assistant")

        # Use SQLite checkpointer in the same data directory
        import os
        from sqlite3 import connect
        os.makedirs("./data", exist_ok=True)
        
        # Initialize SQLite checkpointer with direct connection
        db_path = "./data/supervisor_checkpoints.db"
        conn = connect(db_path, check_same_thread=False)
        checkpointer = SqliteSaver(conn)

        self.agent_graph = builder.compile(checkpointer=checkpointer)

    def get_graph(self):
        """Returns the compiled agent graph"""
        return self.agent_graph

    def display_graph(self):
        """Displays the graph visualization"""
        display(Image(self.agent_graph.get_graph(xray=True).draw_mermaid_png()))

    def invoke(self, messages, config: RunnableConfig = None):
        """
        Invoke the supervisor agent with messages
        
        Args:
            messages: List of messages or single message
            config: RunnableConfig with user_id in configurable
        """
        if not isinstance(messages, list):
            messages = [messages]
        
        # Extract and store user context for tools to use
        if config:
            self.current_user_id = config["configurable"]['user_id']
            self.current_thread_id = config["configurable"]['thread_id']
        
        return self.agent_graph.invoke(
            {"messages": messages},
            config=config
        )

# Create a default instance for backward compatibility
def create_supervisor_agent():
    """Factory function to create a new SupervisorAgent instance"""
    return SupervisorAgent()

# For backward compatibility and direct usage
if __name__ == "__main__":
    supervisor_agent = SupervisorAgent()
    supervisor_agent.display_graph()

