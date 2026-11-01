
import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
import streamlit as st

# Streamlit UI
st.title("🤖 AI QnA Bot")
st.markdown("My QnA Bot with LangChain and NVIDIA Nemotron")

# Load API key 
env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

api_key = os.getenv("NVIDIA_API_KEY")

if not api_key:
    st.error("NVIDIA API key not found!")
    st.stop()

# Create LLM
llm = ChatOpenAI(
    model="nvidia/nemotron-3.5-bolt-e4b",
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=api_key,
    temperature=1.0,
    top_p=0.95
)

# Initialize messages list
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for message in st.session_state.messages:
    role = message["role"]
    content = message["content"]
    st.chat_message(role).markdown(content)

# Take user input
query = st.chat_input("ASK ANYTHING")

if query:

    # Create conversation history
    history = []

    for message in st.session_state.messages:
        history.append({
            "role": message["role"],
            "content": message["content"]
        })

    # Save and display user question
    st.session_state.messages.append({
        "role": "user",
        "content": query
    })

    st.chat_message("user").markdown(query)

    # Add current question to history
    history.append({
        "role": "user",
        "content": query
    })

    # Get AI response
    res = llm.invoke(history)

    # Display AI response
    st.chat_message("assistant").markdown(res.content)

    # Save AI response
    st.session_state.messages.append({
        "role": "assistant",
        "content": res.content
    })
