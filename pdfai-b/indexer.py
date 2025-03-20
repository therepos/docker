import os
import json
import shutil
import traceback
from store import store_in_faiss
from store import initialize_faiss  # Ensure FAISS is initialized properly
from process import chunk_text
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings

# Constants
FAISS_BASE_PATH = "/app"
MODEL_TRACK_FILE = "/app/active_model.txt"
UPLOAD_DIR = "data"
METADATA_FILE = os.path.join(UPLOAD_DIR, "files_metadata.json")

def load_metadata():
    """Loads metadata from a JSON file."""
    if not os.path.exists(METADATA_FILE):
        print("DEBUG: Metadata file not found. No documents uploaded.")
        return {}
    with open(METADATA_FILE, "r") as f:
        return json.load(f)

def list_faiss_indexes():
    """Lists all FAISS indexes currently available."""
    return [folder for folder in os.listdir(FAISS_BASE_PATH) if folder.startswith("faiss_index_")]

def switch_model(new_model: str):
    """Switches to a new model, reindexes data, and deletes the old FAISS index."""
    global OLLAMA_MODEL

    print(f"DEBUG: Initiating model switch to {new_model}...")

    # **Get current FAISS path before switching**
    old_faiss_path = os.getenv("FAISS_INDEX_PATH", f"{FAISS_BASE_PATH}/faiss_index_{OLLAMA_MODEL}")
    print(f"DEBUG: Current FAISS index: {old_faiss_path}")

    # **Switch to the new model**
    OLLAMA_MODEL = new_model
    os.environ["OLLAMA_MODEL"] = new_model

    # **Persist model selection**
    try:
        with open(MODEL_TRACK_FILE, "w") as f:
            f.write(new_model)
        print(f"DEBUG: Saved active model as {new_model}")
    except Exception as e:
        print(f"ERROR: Failed to write active model file: {str(e)}")
        return {"detail": "Failed to save active model selection."}

    # **Create new FAISS index path**
    FAISS_INDEX_PATH = f"{FAISS_BASE_PATH}/faiss_index_{new_model}"
    os.environ["FAISS_INDEX_PATH"] = FAISS_INDEX_PATH  

    # **Ensure directory exists**
    print(f"DEBUG: Creating FAISS index directory: {FAISS_INDEX_PATH}")
    os.makedirs(FAISS_INDEX_PATH, exist_ok=True)

    # **Ensure FAISS is initialized BEFORE deleting the old index**
    print("DEBUG: Ensuring FAISS is initialized for the new model...")
    initialize_faiss()  # This will create an empty FAISS index if none exists

    # **Check if FAISS index was actually created**
    if not os.path.exists(f"{FAISS_INDEX_PATH}/index.faiss"):
        print(f"ERROR: FAISS index missing after switch. Creating an empty FAISS index.")
        initialize_faiss()

    # **Delete the old FAISS index AFTER ensuring the new one exists**
    if os.path.exists(old_faiss_path) and old_faiss_path != FAISS_INDEX_PATH:
        print(f"DEBUG: Deleting old FAISS index: {old_faiss_path}")
        shutil.rmtree(old_faiss_path, ignore_errors=True)

    return {"message": f"Model switched to {new_model}. New FAISS index created."}
