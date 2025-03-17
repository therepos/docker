import os
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings

def store_in_faiss(chunks):
    """Stores extracted text chunks in FAISS vector database using Ollama embeddings."""
    model = os.getenv("OLLAMA_MODEL", "llama3.2")  # Get from env, fallback to "llama3.2"
    embeddings = OllamaEmbeddings(model=model)
    vector_store = FAISS.from_texts(chunks, embeddings)
    vector_store.save_local("faiss_index")
