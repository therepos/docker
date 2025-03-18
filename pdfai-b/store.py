import os
import datetime
import json
import requests
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS

METADATA_FILE = "/app/data/files_metadata.json"  # Ensure consistency with main.py
FAISS_INDEX_PATH = "/app/faiss_index"

def load_metadata():
    """Load existing metadata or return an empty dictionary."""
    print("DEBUG: Loading metadata...")
    if not os.path.exists(METADATA_FILE):
        return {}
    with open(METADATA_FILE, "r") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return {}

def save_metadata(metadata):
    """Save updated metadata to JSON file."""
    print("DEBUG: Saving metadata...")
    with open(METADATA_FILE, "w") as f:
        json.dump(metadata, f, indent=4)

def initialize_faiss(embeddings):
    """Ensure FAISS index is initialized if missing."""
    os.makedirs(FAISS_INDEX_PATH, exist_ok=True)  # Ensure directory exists
    print(f"DEBUG: Checking if FAISS index exists at {FAISS_INDEX_PATH}/index.faiss")

    index_file = f"{FAISS_INDEX_PATH}/index.faiss"
    if not os.path.exists(index_file):
        print("DEBUG: Initializing new FAISS index...")
        vector_store = FAISS.from_texts(["FAISS Initialized"], embeddings, metadatas=[{"uid": "init"}])
        vector_store.save_local(FAISS_INDEX_PATH)

def store_in_faiss(chunks, model, uid):
    """Stores extracted text chunks in FAISS and updates metadata."""
    print(f"DEBUG: Storing {len(chunks)} chunks in FAISS with model: {model}")

    try:
        # Initialize Ollama embedding model
        embeddings = OllamaEmbeddings(model=model)
        print(f"DEBUG: Embedding model {model} initialized")
    except Exception as e:
        print(f"DEBUG: Embedding model creation failed - {str(e)}")
        raise

    # Ensure FAISS is initialized before storing new embeddings
    initialize_faiss(embeddings)

    # Store UID as metadata inside FAISS
    metadatas = [{"uid": uid} for _ in chunks]
    print(f"DEBUG: Metadata for UID {uid}: {metadatas}")

    try:
        print(f"DEBUG: Loading FAISS index from {FAISS_INDEX_PATH}...")
        vector_store = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
        print("DEBUG: Adding chunks to FAISS...")

        # Create request to Ollama API for embeddings
        response = requests.post('http://ollama:11434/api/generate', json={'model': model, 'prompt': chunks})
        
        if response.status_code == 200:
            print("DEBUG: Successfully received embeddings from Ollama")
            response_data = response.json()  # Get full response
            print(f"DEBUG: Full response from Ollama: {response_data}")

            # Add embeddings to FAISS
            vector_store.add_texts(chunks, metadatas=metadatas)
            print(f"DEBUG: Saving FAISS index at {FAISS_INDEX_PATH}...")
            vector_store.save_local(FAISS_INDEX_PATH)
        else:
            print(f"DEBUG: Failed to get embeddings from Ollama: {response.text}")
            raise Exception("Ollama API request failed")

    except Exception as e:
        print(f"DEBUG: FAISS storage failed - {str(e)}")
        raise Exception(f"Error storing embeddings: {str(e)}")

    # Ensure metadata is saved
    metadata = load_metadata()
    metadata[uid] = {
        "model": model,
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "file_size": len(chunks),
    }
    print(f"DEBUG: Updating metadata for UID {uid}")
    save_metadata(metadata)
    print(f"DEBUG: Metadata updated for UID {uid}")
