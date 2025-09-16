#!/usr/bin/env python3

from supervisor_agent import SupervisorAgent
from langchain_core.messages import HumanMessage
from langchain_core.runnables.config import RunnableConfig
import uuid
import sys
import os

def main():
    print("🤖 Supervisor Agent Chat")
    print("Type 'quit' or 'exit' to end the conversation")
    print("=" * 50)
    
    # Ensure data directory exists
    os.makedirs("./data", exist_ok=True)
    
    # Initialize the supervisor agent
    supervisor = SupervisorAgent()
    
    # Generate a unique user ID and thread ID for this session
    user_id = "005"
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

            # user_input = "what all can you do?"

            # Check for exit commands
            if user_input.lower() in ['quit', 'exit', 'bye']:
                print("\n👋 Goodbye!")
                break

            if not user_input:
                continue

            # Create message and invoke supervisor
            message = HumanMessage(content=user_input)
            result = supervisor.invoke([message], config=config)

            # Display agent response
            response = result['messages'][-1].content
            print(f"\n🤖 Assistant: {response}")

        except KeyboardInterrupt:
            print("\n\n👋 Chat interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            print("Please try again or type 'quit' to exit.")

    # user_input = "what all notes are saved for me? please list them"
    #
    # # Check for exit commands
    # if user_input.lower() in ['quit', 'exit', 'bye']:
    #     print("\n👋 Goodbye!")
    #
    # # Create message and invoke supervisor
    # message = HumanMessage(content=user_input)
    # result = supervisor.invoke([message], config=config)
    #
    # # Display agent response
    # response = result['messages'][-1].content
    # print(f"\n🤖 Assistant: {response}")

if __name__ == "__main__":
    main()
