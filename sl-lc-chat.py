# !pip install streamlit langchain langchain-nvidia-ai-endpoints  

import streamlit as st
from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings

# My Variables
myDocs = "./data"

# LLMs and Embeddings
llm = ChatNVIDIA(
    base_url="http://10.106.14.15:8000/v1",
    api_key="FAKE",
    model="meta/llama-3.1-8b-instruct")

#-------------------------------------------------------------
# LangChain Functions
    
def get_response(userInput):
    return llm.invoke(userInput).content

def generate_response(userInput):
    with st.chat_message('human'):
        st.markdown(userInput)
    st.session_state.messages.append({"role": "human", "content": userInput})
    with st.chat_message('assistant'):
        response = get_response(userInput)
        st.write(response)
    st.session_state.messages.append({"role": "assistant", "content": response})

#-------------------------------------------------------------
# Streamlit Stuff
# Config
st.set_page_config(
    page_title = "LangChain Chat")

# Session State
if "messages" not in st.session_state:
    st.session_state.messages = []

# Conversation
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
# Display new Q&A    
userInput = st.chat_input("Ask your question...")
if userInput != None and userInput != "":
    generate_response(userInput)