from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool
import requests
import os
from pathlib import Path

load_dotenv(Path(__file__).parent / ".env")

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen3-32B",
    task="conversational",
    max_new_tokens=512,
    temperature=0.7,
)

model = ChatHuggingFace(llm = llm)

search_tool = DuckDuckGoSearchRun(region="us-en")

@tool
def calculator(first_num: float, second_num: float, operation: str) -> dict:
    """
        Perform basic operation on 2 numbers.
        Supported operations: add, sub,mul, div
    """
    try:
        if operation == "add":
            result = first_num + second_num
        elif operation == "sub":
                    result = first_num - second_num
        elif operation == "mul":
                    result = first_num * second_num
        elif operation == "div":
                    if second_num == 0:
                          return {"error": "Division by zero is not allowed"}
                    result = first_num / second_num
        else:
              return {"error:" f"Unsupported operation{operation}"}
    except Exception as e:
          return {"error": str(e)}

@tool
def get_stock_price(symbol: str) -> dict:
      """Fetch latest stock pricefor a given symbol(e.g. 'AAPL', 'TSLA')
      using Alpha Vantage API key in the URL
      """
      api_key = os.getenv("ALPHAVANTAGE_API_KEY")
      url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey={api_key}"
      r = requests.get(url)
      return r.json()


tools =[search_tool, get_stock_price, calculator]
llm_with_tools = model.bind_tools(tools)
      
          
              
              
    




class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def chat_node(state: ChatState):
    messages=state['messages']
    response = llm_with_tools.invoke(messages)

    return {"messages": [response]}

tool_node = ToolNode(tools)


conn = sqlite3.connect(database='chatbot.db', check_same_thread=False)


# checkpointer
checkpointer = SqliteSaver(conn = conn)

graph = StateGraph(ChatState)

graph.add_node("chat_node", chat_node)
graph.add_node("tools", tool_node)


graph.add_edge(START, 'chat_node')
graph.add_conditional_edges('chat_node', tools_condition )
graph.add_edge('tools', 'chat_node' )

chatbot = graph.compile(checkpointer=checkpointer)

def retrive_all_threads():
    all_threads = set()
    for checkpoint in checkpointer.list(None) :
        all_threads.add(checkpoint.config['configurable']['thread_id'])
    return list(all_threads)


