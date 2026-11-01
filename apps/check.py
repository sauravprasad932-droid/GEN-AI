import sys
from pathlib import Path

# Auto-link project virtualenv site-packages if running with global Python
_venv_site = Path(__file__).resolve().parent.parent / "env" / "Lib" / "site-packages"
if _venv_site.exists() and str(_venv_site) not in sys.path:
    sys.path.insert(0, str(_venv_site))

import os
import warnings
from dotenv import load_dotenv

# Suppress deprecation warnings from langchain_community
warnings.filterwarnings("ignore", category=DeprecationWarning)

from langchain_ollama import ChatOllama
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_community.utilities import GoogleSerperAPIWrapper

# 1. Load environment variables
load_dotenv()

# Check for Serper API key
if not os.getenv("SERPER_API_KEY"):
    raise ValueError("SERPER_API_KEY not found in environment or .env file.")

# 2. Connect LangChain with Ollama
llm = ChatOllama(
    model="qwen2.5:3b",
    temperature=0,
    streaming=True
)

# 3. Define search tool
search_wrapper = GoogleSerperAPIWrapper()

@tool
def google_search(query: str) -> str:
    """Search Google for recent information, facts, news, and answers."""
    return search_wrapper.run(query)

# 4. Create Agent
agent = create_agent(
    model=llm,
    tools=[google_search],
    system_prompt="You are a helpful assistant with access to Google Search. You must always use the google_search tool to search the web for any factual, current, or event-related questions before answering."
)

if __name__ == "__main__":
    # 5. Run query
    question = "ipl 2025 won by which team?"
    print(f"Question: {question}\nThinking & searching...")
    response = agent.invoke({"messages": [{"role": "user", "content": question}]})
    print("\nAnswer:")
    print(response["messages"][-1].content)
