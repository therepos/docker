from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
import datetime

def store_in_faiss(chunks, model, uid):
    """Stores extracted text chunks in FAISS vector database using Ollama embeddings."""
    embeddings = OllamaEmbeddings(model=model)          
    vector_store = FAISS.from_texts(chunks, embeddings)
    vector_store.save_local("/app/faiss_index")         # Ensure path consistency

    # Add UID and associated model information to metadata
    metadata[uid] = {
        "model": model,  # Store the model used for FAISS indexing
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "file_size": len(chunks),  # Example additional metadata
    }