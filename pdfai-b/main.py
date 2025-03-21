import os
import re
import shutil
import random
import string
import zipfile
import json
import traceback
import time
from datetime import datetime
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from extract import extract_text
from store import store_in_faiss
from store import initialize_faiss
from process import chunk_text
from query import query_ai
from indexer import switch_model
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings


# **Constants**
UPLOAD_DIR = "data"
EXPORT_DIR = "/app/export"
FAISS_BASE_PATH = "/app"
MODEL_TRACK_FILE = "/app/active_model.txt"
METADATA_FILE = os.path.join(UPLOAD_DIR, "files_metadata.json")

# **Load Last Used Model on Startup or Set Default**
if os.path.exists(MODEL_TRACK_FILE):
    with open(MODEL_TRACK_FILE, "r") as f:
        OLLAMA_MODEL = f.read().strip()
else:
    OLLAMA_MODEL = "mistral"  # Default model if no previous model is found

# **Ensure OLLAMA_BASE_URL is Set Properly**
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")

# **Set FAISS index path based on active model**
FAISS_INDEX_PATH = f"{FAISS_BASE_PATH}/faiss_index_{OLLAMA_MODEL}"

# **Ensure environment variables are properly set**
os.environ["OLLAMA_BASE_URL"] = OLLAMA_BASE_URL
os.environ["OLLAMA_MODEL"] = OLLAMA_MODEL
os.environ["FAISS_INDEX_PATH"] = FAISS_INDEX_PATH

print(f"DEBUG: Loaded OLLAMA_BASE_URL: {OLLAMA_BASE_URL}")
print(f"DEBUG: Loaded OLLAMA_MODEL: {OLLAMA_MODEL}")
print(f"DEBUG: FAISS Index Path: {FAISS_INDEX_PATH}")

# Ensure FAISS index is initialized on startup
initialize_faiss()

# **Initialize FastAPI App**
app = FastAPI()
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(EXPORT_DIR, exist_ok=True)

# **Helper Functions**
def save_metadata(metadata):
    """Writes metadata safely with a backup."""
    backup_path = METADATA_FILE + ".bak"
    if os.path.exists(METADATA_FILE):
        shutil.copy(METADATA_FILE, backup_path)  # Create backup before saving
    with open(METADATA_FILE, "w") as f:
        json.dump(metadata, f, indent=4)

def load_metadata():
    """Loads metadata from a JSON file."""
    if not os.path.exists(METADATA_FILE):
        return {}
    with open(METADATA_FILE, "r") as f:
        return json.load(f)

def generate_uid():
    """Generates a unique 12-character alphanumeric UID."""
    return ''.join(random.choices(string.ascii_letters + string.digits, k=12))

# **File Management Endpoints**
@app.get("/")
def about():
    """Returns API status."""
    return {"message": "PDF-AI API is running", "version": "1.0", "model": OLLAMA_MODEL}

@app.post("/upload/")
async def upload_file(files: list[UploadFile] = File(...)):
    """Uploads files, extracts text, and stores embeddings in FAISS with metadata tracking."""
    metadata = load_metadata()
    uploaded_files = []

    for file in files:
        uid = generate_uid()
        text_filename = f"{uid}.txt"
        text_path = os.path.join(UPLOAD_DIR, text_filename)

        try:
            file_path = os.path.join(UPLOAD_DIR, file.filename)
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)

            # Read content only once
            if file.filename.endswith(".txt"):
                with open(file_path, "r", encoding="utf-8") as f:
                    extracted_text = f.read()
            else:
                extracted_text = extract_text(file_path)

            if extracted_text.strip():
                with open(text_path, "w", encoding="utf-8") as text_file:
                    text_file.write(extracted_text)

                chunks = chunk_text(extracted_text)
                if chunks:
                    store_in_faiss(chunks, uid)  # Associate chunks with file ID
                    os.remove(file_path)  # Cleanup after processing
                else:
                    print(f"DEBUG: No chunks created for {file.filename}")

                # Store file metadata
                metadata[uid] = {
                    "uid": uid,
                    "original_filename": file.filename,
                    "stored_filename": text_filename,
                    "size_kb": round(len(extracted_text) / 1024, 2),
                    "last_modified": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "model_used": OLLAMA_MODEL
                }
                uploaded_files.append({"file": file.filename, "message": "Text stored in FAISS."})
            else:
                uploaded_files.append({"file": file.filename, "message": "No text extracted."})

        except Exception as e:
            uploaded_files.append({"file": file.filename, "message": f"Error: {str(e)}"})

    save_metadata(metadata)
    return {"message": "Upload complete", "results": uploaded_files}

@app.get("/list_files/")
def list_files():
    """Lists all uploaded files with metadata including the model used."""
    return {"files": load_metadata()}

@app.get("/export_all/")
async def export_all_files():
    """Compresses all uploaded files into a ZIP and returns it."""
    zip_path = os.path.join(EXPORT_DIR, "data_export.zip")
    shutil.make_archive(os.path.join(EXPORT_DIR, "data_export"), 'zip', UPLOAD_DIR)
    return FileResponse(zip_path, media_type="application/zip", filename="data_export.zip")

@app.get("/export_faiss/")
async def export_faiss():
    """Exports FAISS index as a backup file with a timestamp."""
    try:
        timestamp = datetime.now().strftime("%Y%m%d%H%M")
        model_name = OLLAMA_MODEL.replace("/", "_")
        backup_filename = f"faiss_backup_{model_name}_{timestamp}.zip"
        backup_path = os.path.join(EXPORT_DIR, backup_filename)

        shutil.make_archive(backup_path.replace(".zip", ""), 'zip', FAISS_INDEX_PATH)

        return FileResponse(backup_path, media_type="application/zip", filename=backup_filename)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error exporting FAISS index: {str(e)}")

@app.post("/import_faiss/")
async def import_faiss(file: UploadFile = File(...)):
    """Restores FAISS index from a backup file and ensures model consistency."""
    try:
        backup_path = os.path.join(EXPORT_DIR, file.filename)
        with open(backup_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Extract model name from filename
        match = re.search(r"faiss_backup_(.*?)_\d{12}\.zip", file.filename)
        if not match:
            raise HTTPException(status_code=400, detail="Invalid FAISS backup filename format.")

        imported_model = match.group(1)

        # Unpack FAISS backup
        shutil.unpack_archive(backup_path, FAISS_INDEX_PATH, "zip")

        # Check if model matches the active one, if not, switch
        if imported_model != OLLAMA_MODEL:
            switch_model(imported_model)

        return {"message": f"FAISS index for {imported_model} successfully restored."}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error importing FAISS index: {str(e)}")

@app.delete("/delete/all/")
def delete_all_files():
    """Deletes all files, clears FAISS index, and resets everything in memory."""
    try:
        global OLLAMA_MODEL

        # Step 1: Delete All Data Files
        if os.path.exists(UPLOAD_DIR):
            shutil.rmtree(UPLOAD_DIR, ignore_errors=True)
            os.makedirs(UPLOAD_DIR, exist_ok=True)

        # Step 2: Delete Metadata
        if os.path.exists(METADATA_FILE):
            os.remove(METADATA_FILE)

        # Step 3: Delete FAISS Index Completely
        faiss_path = f"/app/faiss_index_{OLLAMA_MODEL}"  # Ensure model-specific FAISS path is used
        if os.path.exists(faiss_path):
            shutil.rmtree(faiss_path, ignore_errors=True)  # Remove entire FAISS directory
            os.makedirs(faiss_path, exist_ok=True)  # Recreate empty FAISS directory

        # Step 4: Force FAISS to reset in memory
        from store import initialize_faiss
        initialize_faiss()  # Reinitialize FAISS with a fresh empty index

        return {"message": "All files, metadata, and FAISS index have been reset."}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error resetting FAISS: {str(e)}")

@app.get("/query/")
def query_extracted_text(question: str, uid: str = None):
    """Queries the extracted text using AI, optionally filtering by file using its uid."""
    try:
        print(f"DEBUG: Query received: {question}")
        print(f"DEBUG: Using model '{OLLAMA_MODEL}' with base URL '{OLLAMA_BASE_URL}'")

        # Load embeddings
        embeddings = OllamaEmbeddings(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL)
        print("DEBUG: Embeddings initialized successfully.")

        # Load FAISS index
        print(f"DEBUG: Attempting to load FAISS index from {FAISS_INDEX_PATH}...")
        vector_store = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
        print("DEBUG: FAISS index loaded successfully.")

        # Retrieve relevant documents
        retriever = vector_store.as_retriever()
        print("DEBUG: FAISS retriever initialized.")

        # If `uid` is provided, filter results to that specific document
        if uid:
            print(f"DEBUG: Filtering results for document UID: {uid}")
            retriever.search_kwargs = {"filter": {"source": uid}}

        # Execute query via `query_ai()`
        response = query_ai(question)

        print(f"DEBUG: Response generated: {response}")
        return {"question": question, "file": uid if uid else "all", "answer": response}

    except Exception as e:
        error_details = traceback.format_exc()
        print(f"ERROR: Query failed - {str(e)}")
        print(f"ERROR DETAILS:\n{error_details}")
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

@app.get("/current_model/")
def get_current_model():
    """Returns the current model being used for FAISS and queries."""
    return {"current_model": OLLAMA_MODEL}

@app.get("/faiss_model/")
def get_active_faiss_model():
    """Returns the FAISS index in use."""
    return {"active_faiss_index": os.getenv("FAISS_INDEX_PATH")}

@app.post("/switch_model/")
def switch_model_endpoint(new_model: str):
    result = switch_model(new_model)

    # Re-sync main.py globals with environment
    global OLLAMA_MODEL, FAISS_INDEX_PATH
    OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", new_model)
    FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH", f"{FAISS_BASE_PATH}/faiss_index_{OLLAMA_MODEL}")

    print(f"DEBUG: Switched model in main.py: {OLLAMA_MODEL}")
    print(f"DEBUG: Updated FAISS path in main.py: {FAISS_INDEX_PATH}")

    return result



