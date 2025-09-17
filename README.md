Now we have a basic terminal app that can manage notes.

To do:
1. Give Donna ability to store user profile
2. Add websearch tools
3. Evals and robustness

Very interesting bug:
Given that I am sending the entire state to notes agent, the state also included 
past conversation history, including the fact that I said ily to donna. It thought that it is 
another note, and said there are 2 notes instead of 1. So Donna thought there are 2 notes instead
of 1. Crazy hallucination.
And now, I asked to delete it, the tool call failed, but the llm still felt that it is deleted 
and came back with a confirmation omg. 