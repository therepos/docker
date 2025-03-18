import os
import datetime
import json
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS

METADATA_FILE = "/app/data/files_metadata.json"  # Ensure consistency with main.py
FAISS_INDEX_PATH = "/app/faiss_index"

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

def initialize_faiss(embeddings):
    """Ensure FAISS index is initialized if missing."""
    os.makedirs(FAISS_INDEX_PATH, exist_ok=True)  # Ensure directory exists

    index_file = f"{FAISS_INDEX_PATH}/index.faiss"
    if not os.path.exists(index_file):
        print("Initializing new FAISS index...")
        vector_store = FAISS.from_texts(["FAISS Initialized"], embeddings, metadatas=[{"uid": "init"}])
        vector_store.save_local(FAISS_INDEX_PATH)

def store_in_faiss(chunks, model, uid):
    """Stores extracted text chunks in FAISS and updates metadata."""
    embeddings = OllamaEmbeddings(model=model)

    # Ensure FAISS is initialized before storing new embeddings
    initialize_faiss(embeddings)

    # Store UID as metadata inside FAISS
    metadatas = [{"uid": uid} for _ in chunks]

    # Load existing FAISS index if available
    vector_store = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
    vector_store.add_texts(chunks, metadatas=metadatas)
    vector_store.save_local(FAISS_INDEX_PATH)

    # Update metadata
    metadata = load_metadata()
    metadata[uid] = {
        "model": model,
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "file_size": len(chunks),
    }
    save_metadata(metadata)  # Save back to file
