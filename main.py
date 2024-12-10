import os

from langchain_openai import ChatOpenAI
from langgraph.constants import START
from langgraph.graph import MessagesState, StateGraph, END
from langchain_core.messages import RemoveMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from dotenv import load_dotenv

load_dotenv()
base_model = os.getenv("BASE_MODEL")
llm = ChatOpenAI(model=base_model, max_retries=2)


class ConversationState(MessagesState):
    summary: str


def call_model(state: ConversationState):
    summary = state.get("summary", "")

    if summary:

        system_message = f"Summary of conversation earlier: {summary}"

        messages = [SystemMessage(content=system_message)] + state["messages"]

    else:
        messages = state['messages']

    response = llm.invoke(messages)

    return {"messages": response}


def summarize_conversation(state: ConversationState) -> ConversationState:

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
    messages = state['messages']

    if len(messages) > 6:
        return "summarize_conversation"

    return END

config = {"configurable": {"thread_id" : "1"}}

workflow = StateGraph(ConversationState)

workflow.add_node("conversation", call_model)
workflow.add_node(summarize_conversation)

workflow.add_edge(START, "conversation")
workflow.add_conditional_edges("conversation", should_continue)
workflow.add_edge("summarize_conversation", END)

memory = MemorySaver()
llm_agent = workflow.compile(checkpointer=memory)

while True:
    user_query = input()
    message = HumanMessage(content=user_query)
    output = llm_agent.invoke({"messages": [message]}, config)
    for m in output['messages'][-1:]:
        m.pretty_print()