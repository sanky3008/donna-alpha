#!/usr/bin/env python3

from supervisor_agent import SupervisorAgent
from langchain_core.messages import HumanMessage
from langchain_core.runnables.config import RunnableConfig

def test_config_fix():
    """Test the fixed config propagation to notes agent"""
    
    print("=== Testing Fixed Config Propagation ===\n")
    
    supervisor = SupervisorAgent()
    
    # Test with specific user
    config = RunnableConfig(
        configurable={
            "user_id": "test_user_123",
            "thread_id": "test_session",
            "checkpoint_ns": "test_user_123"
        }
    )
    
    print("1. Creating a note with user isolation:")
    message1 = HumanMessage(content="Create a note that says 'This note should have proper user metadata'")
    result1 = supervisor.invoke([message1], config=config)
    print(f"Response: {result1['messages'][-1].content}\n")
    
    print("2. Creating another note for the same user:")
    message2 = HumanMessage(content="Create another note saying 'Second note with user isolation'") 
    result2 = supervisor.invoke([message2], config=config)
    print(f"Response: {result2['messages'][-1].content}\n")
    
    # Test with different user
    config2 = RunnableConfig(
        configurable={
            "user_id": "different_user_456", 
            "thread_id": "different_session",
            "checkpoint_ns": "different_user_456"
        }
    )
    
    print("3. Creating a note for a different user:")
    message3 = HumanMessage(content="Create a note that says 'This is from a different user'")
    result3 = supervisor.invoke([message3], config=config2)
    print(f"Response: {result3['messages'][-1].content}\n")
    
    print("✅ Config fix test completed!")

if __name__ == "__main__":
    test_config_fix()