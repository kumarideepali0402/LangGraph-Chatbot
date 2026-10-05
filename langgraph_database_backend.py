from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-Coder-7B-Instruct",
    task="text-generation",
    max_new_tokens=512,
    temperature=0.7,
)

model = ChatHuggingFace(llm = llm)


from langgraph.graph.message import add_messages

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def chat_node(state: ChatState):
    messages=state['messages']
    response = model.invoke(messages)

    return {"messages": [response]}


conn = sqlite3.connect(database='chatbot.db', check_same_thread=False)


# checkpointer
checkpointer = SqliteSaver(conn = conn)

graph = StateGraph(ChatState)

graph.add_node("chat_node", chat_node)


graph.add_edge(START, 'chat_node')
graph.add_edge('chat_node', END )

chatbot = graph.compile(checkpointer=checkpointer)

def retrive_all_threads():
    all_threads = set()
    for checkpoint in checkpointer.list(None) :
        all_threads.add(checkpoint.config['configurable']['thread_id'])
    return list(all_threads)


