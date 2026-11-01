
import sys
from pathlib import Path

# 1. Auto-link project virtualenv site-packages
_venv_site = Path(__file__).resolve().parent.parent / "env" / "Lib" / "site-packages"

if _venv_site.exists() and str(_venv_site) not in sys.path:
    sys.path.insert(0, str(_venv_site))

import os
import warnings
from dotenv import load_dotenv

# 2. Suppress deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from langchain_ollama import ChatOllama
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_community.utilities import GoogleSerperAPIWrapper
from langgraph.checkpoint.memory import InMemorySaver


# 3. Load environment variables
load_dotenv()

# 4. Validate Serper API Key
if not os.getenv("SERPER_API_KEY"):
    raise ValueError("SERPER_API_KEY not found in environment or .env file.")

# 5. Connect Ollama LLM
llm = ChatOllama(
    model="qwen2.5:3b",
    temperature=0,
    streaming=True
)

# 6. Initialize Google Search
search = GoogleSerperAPIWrapper()

# 7. Create Google Search Tool
@tool
def google_search(query: str) -> str:
    """Search Google for current information, facts, news and answers."""
    return search.run(query)

# 8. Create Agent
agent = create_agent(
    model=llm,
    tools=[google_search],
    checkpointer=InMemorySaver(),
    system_prompt="""
    You are a helpful AI search agent.

    Whenever the user asks a factual, current,
    or event-related question, use google_search.

    Answer using search results.
    Do not invent information.
    """
)

# 9. Run Agent
if __name__ == "__main__":
    while True:
        
        question = input("ASK QUESTION -> ")
        if question.lower()=="quit":
            print("BYE 👋")
            break
        print(f"Question: {question}")
        print("Thinking and searching...")

        response = agent.invoke(
            {"messages": [{"role": "user", "content": question}]},
            config={"configurable": {"thread_id": "1"}},
        )
        # 10. Print Final Answer
        print("\nFinal Answer:")
        print("AI -> ", response["messages"][-1].content)
