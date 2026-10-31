
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
import streamlit as st

# Load API key
load_dotenv()

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
prompts=[
    {"role":"user","content":"Hello, My name is prasad"},
    {"role":"ai","content":"hello saurav ! hoe can i assit you"},
    {"role":"user","content":"what is my name"}
]
res=llm.invoke(prompts)
print(res.content)
from langchain_core.messages import HumanMessage, AIMessage

history = []

while True:
    query = input("User -> ")

    if query.lower() in ["exit", "quit", "bye"]:
        print("Good Bye 👋")
        break

    history.append(HumanMessage(content=query))

    print("User_question:", query)

    res = llm.invoke(history)

    history.append(AIMessage(content=res.content))

    print("AI ->", res.content, "\n")
