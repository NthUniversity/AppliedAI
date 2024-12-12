# !pip install streamlit langchain langchain-nvidia-ai-endpoints langchain_community faiss-cpu pypdf

import streamlit as st
from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
import os

# My Variables
myDocs = "./data"

# LLMs and Embeddings
llm = ChatNVIDIA(
    base_url="http://10.106.14.15:8000/v1",
    api_key="FAKE",
    model="meta/llama-3.1-8b-instruct")

embedding = NVIDIAEmbeddings(
    base_url="http://10.106.14.18:8031/v1",
    api_key="FAKE",
    model="nvidia/nv-embedqa-e5-v5",
    )

#-------------------------------------------------------------
# LangChain Workflow:
# Get VectorStore
# build context retriever chain (from Vectorstore)
# build conversational RAG chain (from retriever chain)
# get response 
#-------------------------------------------------------------
# LangChain Functions

def get_vectorstore():
    loader = PyPDFDirectoryLoader(myDocs)
    documents = loader.load()

    # Chunk the data
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", ";", ",", " ", ""],
    )
    docChunks = text_splitter.split_documents(documents)
    
    # Create VectorDB
    vectorStore = FAISS.from_documents(docChunks, embedding=embedding)
    return vectorStore

def get_context_retriever_chain(vectorStore):
    retriever = vectorStore.as_retriever()
    prompt = ChatPromptTemplate.from_messages([
      MessagesPlaceholder(variable_name="chat_history"),
      ("user", "{input}"),
      ("user", """Given the above conversation, generate a search query to 
       look up in order to get information relevant to the conversation""")
    ])
    retrieverChain = create_history_aware_retriever(llm, retriever, prompt)
    return retrieverChain

def get_conversational_rag_chain(retrieverChain): 
    prompt = ChatPromptTemplate.from_messages([
      ("system", "Answer the user's questions based on the below context:\n\n{context}"),
      MessagesPlaceholder(variable_name="chat_history"),
      ("user", "{input}"),
    ])
    stuffDocumentsChain = create_stuff_documents_chain(llm,prompt)
    return create_retrieval_chain(retrieverChain, stuffDocumentsChain)
    
def get_response(userInput):
    retriever_chain = get_context_retriever_chain(st.session_state.vectorStore)
    conversation_rag_chain = get_conversational_rag_chain(retriever_chain)
    response = conversation_rag_chain.invoke({
        "chat_history": st.session_state.messages,
        "input": userInput
    })
    #return response
    responseString = response['answer']
    for doc in response['context']:
        responseString += '\n\n'
        responseString += doc.metadata['source']
        responseString += ' Page '
        responseString += str(doc.metadata['page'])
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
    page_title = "LangChain RAG + Metadata")

# Session State
if "vectorStore" not in st.session_state:
        st.session_state.vectorStore = get_vectorstore()
  
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