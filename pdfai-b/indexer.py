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
    global OLLAMA_MODEL

    print(f"DEBUG: Switching model to {new_model}...")

    # **Get current FAISS path before switching**
    old_faiss_path = os.getenv("FAISS_INDEX_PATH", f"{FAISS_BASE_PATH}/faiss_index_{OLLAMA_MODEL}")
    print(f"DEBUG: Current FAISS index: {old_faiss_path}")

    # **List existing FAISS indexes before switching**
    faiss_before = list_faiss_indexes()
    print(f"DEBUG: Existing FAISS indexes before switch: {faiss_before}")

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

    # **Ensure directory exists**
    print(f"DEBUG: Creating FAISS index directory: {new_faiss_path}")
    try:
        os.makedirs(new_faiss_path, exist_ok=True)
    except Exception as e:
        print(f"ERROR: Failed to create FAISS directory: {str(e)}")
        return {"detail": "FAISS index creation failed: Directory error."}

    # **Load metadata to get stored text files**
    metadata = load_metadata()
    text_chunks = []
    print(f"DEBUG: Loaded metadata with {len(metadata)} entries.")

    for uid, item in metadata.items():
        text_file_path = os.path.join(UPLOAD_DIR, item["stored_filename"])
        if os.path.exists(text_file_path):
            print(f"DEBUG: Processing file {text_file_path} for reindexing...")
            try:
                with open(text_file_path, "r", encoding="utf-8") as text_file:
                    text = text_file.read()
                chunks = chunk_text(text)
                text_chunks.extend(chunks)
                print(f"DEBUG: Extracted {len(chunks)} chunks from {item['stored_filename']}")
            except Exception as e:
                print(f"ERROR: Failed to process {text_file_path}: {str(e)}")
        else:
            print(f"WARNING: File {text_file_path} not found, skipping.")

    print(f"DEBUG: Total text chunks collected for FAISS re-indexing: {len(text_chunks)}")

    # **Check if there are chunks to index**
    try:
        print("DEBUG: Initializing FAISS embeddings and storing data...")
        embeddings = OllamaEmbeddings(model=new_model, base_url="http://ollama:11434")

        if text_chunks:
            faiss_index = FAISS.from_texts(text_chunks, embeddings)
        else:
            faiss_index = FAISS.from_texts(["FAISS RESET"], embeddings)  # Prevent empty FAISS error

        faiss_index.save_local(new_faiss_path)
        print(f"DEBUG: FAISS index successfully saved at {new_faiss_path}")

        # **Update FAISS memory to clear cache**
        global FAISS_INDEX
        FAISS_INDEX = FAISS.load_local(new_faiss_path, embeddings)
        print("DEBUG: FAISS memory cleared and reloaded with new model.")

    except Exception as e:
        print(f"ERROR: Failed to create FAISS index: {str(e)}")
        print(traceback.format_exc())
        shutil.rmtree(new_faiss_path, ignore_errors=True)
        return {"detail": f"FAISS index creation failed: {str(e)}"}

    # **Delete the old FAISS index since it's no longer used**
    if os.path.exists(old_faiss_path):
        print(f"DEBUG: Deleting old FAISS index: {old_faiss_path}")
        shutil.rmtree(old_faiss_path, ignore_errors=True)

    return {"message": f"Model switched to {new_model}. New FAISS index created, loaded into memory, and old index deleted."}
