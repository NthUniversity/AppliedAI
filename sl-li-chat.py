# !pip install streamlit llama-index llama-index-llms-nvidia llama-index-embeddings-nvidia

import streamlit as st
from llama_index.llms.nvidia import NVIDIA 

# LLMs and Embeddings
llm = NVIDIA(
    base_url="http://10.106.14.18:8021/v1",
    api_key="FAKE",
    model="meta/llama-3.1-8b-instruct")

# Streamlit Functions

def get_response(userInput):
    response = llm.complete(userInput).text
    return response

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
    page_title = "LlamaIndex Chat")

# Session State
if "messages" not in st.session_state:
    st.session_state.messages = []

# Conversation
# Display chat history form session history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
# Display new Q&A    
userInput = st.chat_input("Ask your question...")
if userInput != None and userInput != "":
    generate_response(userInput)


