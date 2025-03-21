from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
import os

# Fetch model from environment variable
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")  # Default to llama3.2

def store_in_faiss(chunks):
    """Stores extracted text chunks in FAISS vector database using Ollama embeddings."""
    embeddings = OllamaEmbeddings(model=OLLAMA_MODEL)  # Ensure consistency with querying
    vector_store = FAISS.from_texts(chunks, embeddings)
    vector_store.save_local("/app/faiss_index")  # Ensure path consistency
