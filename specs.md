# Scope

This version is meant for the user to simply store notes with the agent. The agent will also have a running memory of the users' preferences, so that it can be personalised better.

## Job Stories

1. While chatting with the agent, I want the agent to help me with normal question answer as well
2. In case it is something that the agent should remember (like a note), it should store it and use it in the future when the topic comes up
3. Once quite a lot of notes are accumulated, the agent should allow me to update or delete some of them
4. As I chat with the agent, I want the agent to also know and remember what I like and don't like, so that it is personalised for me

## Constraints

1. Completely local setup
2. Terminal chat/minimal UI

## Functional Requirements (FRs)

1. Normal LLM integration - gpt-5-nano or ollama
2. Chat interface via terminal or streamlit
3. Supervisor Agent with profile storage - pos
4. Notes agent with notes storage
5. Persistent storage for all chats

## Non-Functional Requirements (NFRs)

1. All stores should be persistent - can start with local
2. The design should be scalable to multiple platform messages - WA, Meta, etc.
3. The design should handle multiple users without memory leaks

## Agent Design

![Agent Design](/image.png)

### User Profile Store

| Name | Description |
|------|-------------|
| namespace | Logical grouping (e.g. "user_profile") |
| key | Usually "user_{id}" |
| value | JSON with preferences & facts |

### Notes Store

| Name | Description |
|------|-------------|
| text | note |
| embedding | embedding of the note |
| metadata | user_id |

### Checkpoints Store

| Name | Description |
|------|-------------|
| thread_id | Unique conversation ID → channel (whatsapp, terminal, etc.) |
| checkpoint_ns | Namespace for memory separation → user ID |
| checkpoint_id | Turn number / checkpoint |
| data (JSON) | Full agent state at that point |
