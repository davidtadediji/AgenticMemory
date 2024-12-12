import os

from PIL import Image as PILImage
from langchain_openai import ChatOpenAI
from langgraph.constants import START
from langgraph.graph import MessagesState, StateGraph, END
from langchain_core.messages import RemoveMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from dotenv import load_dotenv

import sqlite3

db_path = "state_db/experiment.db"

conn = sqlite3.connect(db_path, check_same_thread=False)

load_dotenv()
base_model = os.getenv("BASE_MODEL")
llm = ChatOpenAI(model=base_model, max_retries=2, max_tokens=80, temperature=1)


class ConversationState(MessagesState):
    summary: str


def call_model(state: ConversationState):
    """
    Process the current conversation state by invoking the language model.
    """
    summary = state.get("summary", "")

    if summary:

        system_message = f"Summary of conversation earlier: {summary}"

        messages = [SystemMessage(content=system_message)] + state["messages"]

    else:
        messages = state["messages"]

    response = llm.invoke(messages)

    return {"messages": response}


def summarize_conversation(state: ConversationState) -> ConversationState:
    """
    Generate or update a summary of the current conversation."""

    summary = state.get("summary", "")

    if summary:

        summary_message = (
            f"This is a summary of the conversation to date: {summary}\n\n"
            "Extend the summary by taking into account the new messages above:"
        )

    else:
        summary_message = "Create a summary of the conversation above:"

    messages = state["messages"] + [HumanMessage(content=summary_message)]

    response = llm.invoke(messages)

    delete_messages = [RemoveMessage(id=m.id) for m in state["messages"][:-2]]

    return {"summary": str(response.content), "messages": delete_messages}


def should_continue(state: ConversationState):
    """
    Determine whether the conversation should continue or move to summarization.
    """
    messages = state["messages"]

    if len(messages) > 6:
        return "summarize_conversation"

    return END


config = {"configurable": {"thread_id": "1"}}

workflow = StateGraph(ConversationState)

workflow.add_node("conversation", call_model)
workflow.add_node(summarize_conversation)

workflow.add_edge(START, "conversation")
workflow.add_conditional_edges("conversation", should_continue)
workflow.add_edge("summarize_conversation", END)

memory = SqliteSaver(conn)
llm_agent = workflow.compile(checkpointer=memory)


# Display graph
image_data = llm_agent.get_graph().draw_mermaid_png()

from pathlib import Path

# Get the absolute path to the project root or app directory
project_root = (
    Path(__file__).resolve().parent
)  # This is the directory where server.py is located
resources_dir = (
    project_root / "resources"
)  # Use the / operator to join paths (pathlib feature)

# Ensure the resources directory exists
resources_dir.mkdir(parents=True, exist_ok=True)

graph_image = resources_dir / "graph.png"  # Creates an absolute path to the file


with open(graph_image, "wb") as f:
    f.write(image_data)


img = PILImage.open("resources/graph.png")
img.show()




while True:
    user_query = input()
    if user_query == "get_prev":
        state_snapshot = llm_agent.get_state(config)
        for value in state_snapshot.values['messages']:
            value.pretty_print()
    else:
        message = HumanMessage(content=user_query)
        output = llm_agent.invoke({"messages": [message]}, config)
        for message in output["messages"][-1:]:
            message.pretty_print()


