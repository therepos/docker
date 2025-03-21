import os
import json
import shutil
import traceback
from store import store_in_faiss, initialize_faiss
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
    print(f"DEBUG: Initiating model switch to {new_model}...")

    # Get old model from environment (fallback: mistral)
    old_model = os.getenv("OLLAMA_MODEL", "mistral")
    old_faiss_path = os.getenv("FAISS_INDEX_PATH", f"{FAISS_BASE_PATH}/faiss_index_{old_model}")
    print(f"DEBUG: Current FAISS index: {old_faiss_path}")

    # List existing FAISS indexes before switching
    faiss_before = list_faiss_indexes()
    print(f"DEBUG: Existing FAISS indexes before switch: {faiss_before}")

    # Update environment variables to new model
    os.environ["OLLAMA_MODEL"] = new_model
    new_faiss_path = f"{FAISS_BASE_PATH}/faiss_index_{new_model}"
    os.environ["FAISS_INDEX_PATH"] = new_faiss_path

    # Save model selection to disk
    try:
        with open(MODEL_TRACK_FILE, "w") as f:
            f.write(new_model)
        print(f"DEBUG: Saved active model as {new_model}")
    except Exception as e:
        print(f"ERROR: Failed to write active model file: {str(e)}")
        return {"detail": "Failed to save active model selection."}

    # Ensure FAISS directory exists
    print(f"DEBUG: Creating FAISS index directory: {new_faiss_path}")
    os.makedirs(new_faiss_path, exist_ok=True)

    # Load metadata and collect chunks
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
                tagged_chunks = [{"uid": uid, "text": chunk} for chunk in chunks]
                text_chunks.extend(tagged_chunks)
                print(f"DEBUG: Extracted {len(chunks)} chunks from {item['stored_filename']}")
            except Exception as e:
                print(f"ERROR: Failed to process {text_file_path}: {str(e)}")
        else:
            print(f"WARNING: File {text_file_path} not found, skipping.")

    print(f"DEBUG: Total text chunks collected for FAISS re-indexing: {len(text_chunks)}")

    # If no chunks, just initialize empty FAISS and return
    if not text_chunks:
        print("WARNING: No chunks found. Creating empty FAISS index.")
        initialize_faiss()
        return {"message": f"Model switched to {new_model}, but no text to index."}

    # Reindex with new embeddings
    try:
        print("DEBUG: Initializing FAISS embeddings and storing data...")
        embeddings = OllamaEmbeddings(model=new_model, base_url="http://ollama:11434")
        texts = [item["text"] for item in text_chunks]
        metadatas = [{"source": item["uid"]} for item in text_chunks]
        faiss_index = FAISS.from_texts(texts, embeddings, metadatas=metadatas)
        faiss_index.save_local(new_faiss_path)
        print(f"DEBUG: FAISS index successfully saved at {new_faiss_path}")
    except Exception as e:
        print(f"ERROR: Failed to create FAISS index: {str(e)}")
        print(traceback.format_exc())
        shutil.rmtree(new_faiss_path, ignore_errors=True)
        return {"detail": f"FAISS index creation failed: {str(e)}"}

    # List FAISS indexes after switch
    faiss_after = list_faiss_indexes()
    print(f"DEBUG: Existing FAISS indexes after switch: {faiss_after}")

    # Delete old FAISS if different
    if old_faiss_path != new_faiss_path and os.path.exists(old_faiss_path):
        print(f"DEBUG: Deleting old FAISS index: {old_faiss_path}")
        shutil.rmtree(old_faiss_path, ignore_errors=True)

    return {"message": f"Model switched to {new_model}. New FAISS index created and reindexed."}
