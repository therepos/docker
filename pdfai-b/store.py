import os
import datetime
import json
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS

METADATA_FILE = "/app/data/files_metadata.json"  # Ensure consistency with main.py

def load_metadata():
    """Load existing metadata or return an empty dictionary."""
    if not os.path.exists(METADATA_FILE):
        return {}
    with open(METADATA_FILE, "r") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return {}

def save_metadata(metadata):
    """Save updated metadata to JSON file."""
    with open(METADATA_FILE, "w") as f:
        json.dump(metadata, f, indent=4)

def store_in_faiss(chunks, model, uid):
    """Stores extracted text chunks in FAISS and updates metadata."""
    embeddings = OllamaEmbeddings(model=model)          
    vector_store = FAISS.from_texts(chunks, embeddings)
    vector_store.save_local("/app/faiss_index")

    # Load metadata before updating
    metadata = load_metadata()
    metadata[uid] = {
        "model": model,
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "file_size": len(chunks),
    }
    save_metadata(metadata)  # Save back to file
