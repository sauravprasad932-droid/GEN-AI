
## LLM
## TOOL-GOOGLE SEARCH TOOL
## AGENT
## MEMORY
## STREAMING
## WEB DEVELOPMENT

import sys
from pathlib import Path

# 1. Auto-link project virtualenv site-packages
_venv_site = Path(__file__).resolve().parent.parent / "env" / "Lib" / "site-packages"

if _venv_site.exists() and str(_venv_site) not in sys.path:
    sys.path.insert(0, str(_venv_site))

import os
import warnings
import streamlit as st
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

# 5. Initialize Streamlit history
if "history" not in st.session_state:
    st.session_state.history = []

if "memory" not in st.session_state:
    st.session_state.memory = InMemorySaver()

# 6. Connect Ollama LLM
llm = ChatOllama(
    model="qwen2.5:3b",
    temperature=0,
    streaming=True
)

# 7. Initialize Google Search
search = GoogleSerperAPIWrapper()

@tool
def google_search(query: str) -> str:
    """Search Google for information and latest news."""
    return search.run(query)

tools = [google_search]

# 8. Create Agent
agent = create_agent(
    model=llm,
    tools=tools,
    checkpointer=st.session_state.memory,
    system_prompt="You are an amazing AI agent and can search on Google as well."
)

# 9. Streamlit UI
st.title("🤖 QuickAnswer")
st.subheader("Answer at the speed of thought")

# 10. Display Previous History
for message in st.session_state.history:
    st.chat_message(message["role"]).markdown(message["content"])

# 11. User Input
query = st.chat_input("Ask Anything?")

if query:

    # Display User Message
    st.chat_message("user").markdown(query)

    st.session_state.history.append({
        "role": "user",
        "content": query
    })

    # Agent Configuration
    config = {
        "configurable": {
            "thread_id": "1"
        }
    }

    # 12. Agent Streaming
    response = agent.stream(
        {
            "messages": [
                {"role": "user", "content": query}
            ]
        },
        config=config,
        stream_mode="messages"
    )

    # 13. Display AI Response
    ai_container = st.chat_message("assistant")

    with ai_container:

        space = st.empty()
        message = ""

        for chunk, metadata in response:

            # Ignore tool messages; display only LLM output
            if metadata.get("langgraph_node") == "model":

                if isinstance(chunk.content, str):
                    message += chunk.content
                    space.markdown(message)

        # 14. Save AI Response
        st.session_state.history.append({
            "role": "assistant",
            "content": message
        })