from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
import os

# Fetch model from environment variable
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "mistral")

def store_in_faiss(chunks):
    """Stores extracted text chunks in FAISS vector database."""
    embeddings = OllamaEmbeddings(model=OLLAMA_MODEL)
    vector_store = FAISS.from_texts(chunks, embeddings)
    vector_store.save_local("/app/faiss_index")
