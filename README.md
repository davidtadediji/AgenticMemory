# Conversation Workflow with LangGraph and LangChain

This project demonstrates a conversational state machine built using `LangGraph` and `LangChain` for managing and summarizing conversations with a conversational AI model. The workflow incorporates memory checkpointing and dynamic state transitions.

## Features

- **State Management:** Tracks conversation state using `MessagesState` and a state graph (`StateGraph`).
- **Conversation Summarization:** Automatically summarizes long conversations for better context retention.
- **Dynamic Workflow:** Handles conditional transitions based on the length of messages.
- **Integration with OpenAI Models:** Uses `ChatOpenAI` for generating conversational responses.
- **Memory Checkpoints:** Saves intermediate states for efficient execution and debugging.

---

## Prerequisites

- Python 3.8+
- Environment variables configured via a `.env` file:
  - `BASE_MODEL`: Specify the OpenAI model (e.g., `gpt-4`).

Install the required Python libraries:
```bash
pip install langchain-openai langgraph python-dotenv
