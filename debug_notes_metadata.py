#!/usr/bin/env python3

from notes_agent import NotesAgent
from langchain_core.messages import HumanMessage
from langchain_core.runnables.config import RunnableConfig

def debug_notes_metadata():
    """Debug why metadata isn't being stored"""
    
    print("=== Debugging Notes Metadata Storage ===\n")
    
    notes_agent = NotesAgent()
    
    # Add debug to the create_note method
    original_create_note = notes_agent.create_note
    
    def debug_create_note(note: str):
        print(f"DEBUG create_note called with:")
        print(f"  note: {note}")
        print(f"  current_user_id: {notes_agent.current_user_id}")
        result = original_create_note(note)
        print(f"  result: {result}")
        return result
    
    notes_agent.create_note = debug_create_note
    notes_agent.tools = [notes_agent.create_note, notes_agent.read_note, notes_agent.update_note, notes_agent.delete_note, notes_agent.get_all_notes]
    notes_agent.llm_with_tools = notes_agent.llm.bind_tools(notes_agent.tools)
    
    config = RunnableConfig(
        configurable={
            "user_id": "debug_metadata_user",
            "thread_id": "debug_thread",
            "checkpoint_ns": "debug_metadata_user"
        }
    )
    
    print("Testing direct notes agent with debug...")
    message = HumanMessage(content="Create a note that says 'Debug metadata test'")
    result = notes_agent.invoke([message], config=config)
    print(f"Final result: {result['messages'][-1].content}")

if __name__ == "__main__":
    debug_notes_metadata()