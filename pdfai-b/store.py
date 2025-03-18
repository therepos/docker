import os
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS

# Fetch model and base URL from environment variables
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "mistral")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")

def store_in_faiss(chunks):
    """Stores extracted text chunks in FAISS vector database."""
    embeddings = OllamaEmbeddings(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL)
    vector_store = FAISS.from_texts(chunks, embeddings)
    vector_store.save_local("/app/faiss_index")  # Ensures path consistency
