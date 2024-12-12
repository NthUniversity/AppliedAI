# !pip install streamlit llama-index llama-index-llms-nvidia llama-index-embeddings-nvidia

import streamlit as st
from llama_index.llms.nvidia import NVIDIA 
from llama_index.embeddings.nvidia import NVIDIAEmbedding
from llama_index.core import Settings, VectorStoreIndex, SimpleDirectoryReader

# My Variables
myDocs = "./data"

# LLMs and Embeddings
llm = NVIDIA(
    base_url="http://10.106.14.18:8021/v1",
    api_key="FAKE",
    model="meta/llama-3.1-8b-instruct")

embedding = NVIDIAEmbedding(
    base_url="http://10.106.14.18:8031/v1",
    api_key="FAKE",
    model="nvidia/nv-embedqa-e5-v5",
    truncate="END"
    )
#-------------------------------------------------------------
# LlamaIndex Workflow Stuff
# Settings
Settings.llm = llm
Settings.embed_model = embedding

# Load
documents = SimpleDirectoryReader(myDocs).load_data()

# Index
index = VectorStoreIndex.from_documents(documents)

# Store
# Running in-memory only

# Query
query_engine = index.as_query_engine()

# Streamlit Functions
def get_response(userInput):
    response = query_engine.query(userInput)
    responseString = response.response
    return responseString

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
    page_title = "LlamaIndex RAG")

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

