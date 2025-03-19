import os
import json
import shutil
import traceback
from store import store_in_faiss
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
    """Switches to a new model, reindexes data, and resets FAISS cache."""
    global OLLAMA_MODEL, FAISS_INDEX

    print(f"DEBUG: Switching model to {new_model}...")

    # **Get current FAISS path before switching**
    old_faiss_path = os.getenv("FAISS_INDEX_PATH", f"{FAISS_BASE_PATH}/faiss_index_{OLLAMA_MODEL}")

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
    new_faiss_path = f"{FAISS_BASE_PATH}/faiss_index_{new_model}"
    os.environ["FAISS_INDEX_PATH"] = new_faiss_path  

    try:
        os.makedirs(new_faiss_path, exist_ok=True)

        # **Load metadata to get stored text files**
        metadata = load_metadata()
        text_chunks = []

        for uid, item in metadata.items():
            text_file_path = os.path.join(UPLOAD_DIR, item["stored_filename"])
            if os.path.exists(text_file_path):
                with open(text_file_path, "r", encoding="utf-8") as text_file:
                    text_chunks.extend(chunk_text(text_file.read()))

        # **Create and save new FAISS index**
        embeddings = OllamaEmbeddings(model=new_model, base_url="http://ollama:11434")

        if text_chunks:
            faiss_index = FAISS.from_texts(text_chunks, embeddings)
        else:
            faiss_index = FAISS.from_texts(["FAISS RESET"], embeddings)

        faiss_index.save_local(new_faiss_path)

        # **Delete the old FAISS index**
        if os.path.exists(old_faiss_path):
            shutil.rmtree(old_faiss_path, ignore_errors=True)
            print(f"DEBUG: Deleted old FAISS index at {old_faiss_path}")

        # **Force FAISS to reload into memory**
        FAISS_INDEX = FAISS.load_local(new_faiss_path, embeddings)
        print("DEBUG: FAISS memory cleared and reloaded with new model.")

        return {"message": f"Model switched to {new_model}. New FAISS index created, loaded into memory, and old index deleted."}

    except Exception as e:
        print(f"ERROR: Failed to switch model: {str(e)}")
        return {"detail": f"Error switching model: {str(e)}"}

