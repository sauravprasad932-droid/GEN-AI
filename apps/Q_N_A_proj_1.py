
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
import streamlit as st

# Streamlit UI
st.title("🤖 AI QnA Bot")
st.markdown("My QnA Bot with LangChain and NVIDIA Nemotron")

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
# initalise the messages list to store the chats
if "messages" not in st.session_state:
    st.session_state.messages=[]

# for displaying the messages
for message in st.session_state.messages:
    role=message["role"]
    content=message["content"]
    st.chat_message(role).markdown(content)
    
query=st.chat_input("ASK ANYTHING")

history = []
if query:
    #save the user input
    st.session_state.messages.append({
        "role":"user",
        "content":query})
    #display the user input
    st.chat_message("user").markdown(query)
    
    #getting response from llm
    res=llm.invoke(query)
    
    #display the ai messsage
    st.chat_message("assistant").markdown(res.content)
    
    #save AI response
    st.session_state.messages.append({
        "role":"assistant",
        "content":res.content
    })
    
    
history = []

while True:
    query = input("User -> ")

    if query.lower() in ["exit", "quit", "bye"]:
        print("Good Bye 👋")
        break

    history.append(HumanMessage(content=query))
    # display and save the user question
    st.session_state.messages.append({
        "role":"user",
        "content":query})
    #display the user input
    st.chat_message("user").markdown(query)
    

    res = llm.invoke(history)
     #display the ai messsage
    st.chat_message("assistant").markdown(res.content)
    
    #save AI response
    st.session_state.messages.append({
        "role":"assistant",
        "content":res.content
    })

    history.append(AIMessage(content=res.content))

    print("AI ->", res.content, "\n")
    