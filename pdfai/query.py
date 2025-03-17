import os
import requests
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.llms import Ollama
from langchain_ollama import OllamaEmbeddings

# Fetch Ollama host and model from environment variables
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://ollama:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")  # Default to llama3.2

def load_vector_store():
    """Loads stored text chunks from FAISS using the correct embeddings."""
    return FAISS.load_local("/app/faiss_index", OllamaEmbeddings(model=OLLAMA_MODEL), allow_dangerous_deserialization=True)

def query_ai(query):
    """Retrieves relevant text and generates an AI answer using Ollama."""
    retriever = load_vector_store().as_retriever()

    if not OLLAMA_MODEL:
        return "No available models found in Ollama."

    llm = Ollama(model=OLLAMA_MODEL)
    qa_chain = RetrievalQA.from_chain_type(llm, retriever=retriever)
    
    return qa_chain.run(query)
