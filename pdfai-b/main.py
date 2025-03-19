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
from process import chunk_text
from query import query_ai
from indexer import switch_model
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings

# Constants
UPLOAD_DIR = "data"
EXPORT_DIR = "/app/export"
FAISS_BASE_PATH = "/app"
MODEL_TRACK_FILE = "/app/active_model.txt"
METADATA_FILE = os.path.join(UPLOAD_DIR, "files_metadata.json")

# **Load Last Used Model on Startup**
if os.path.exists(MODEL_TRACK_FILE):
    with open(MODEL_TRACK_FILE, "r") as f:
        OLLAMA_MODEL = f.read().strip()
else:
    OLLAMA_MODEL = "mistral"  # Default model

# Set FAISS path based on active model
os.environ["OLLAMA_MODEL"] = OLLAMA_MODEL
os.environ["FAISS_INDEX_PATH"] = f"{FAISS_BASE_PATH}/faiss_index_{OLLAMA_MODEL}"
print(f"DEBUG: Loaded last used model: {OLLAMA_MODEL}")

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
    """Uploads files, extracts text, and stores embeddings in FAISS."""
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

            extracted_text = file.read().decode("utf-8") if file.filename.endswith(".txt") else extract_text(file_path)

            if extracted_text.strip():
                with open(text_path, "w", encoding="utf-8") as text_file:
                    text_file.write(extracted_text)

                chunks = chunk_text(extracted_text)
                store_in_faiss(chunks)
                os.remove(file_path)

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
        save_model_label()
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
        # Save uploaded file
        backup_path = os.path.join(EXPORT_DIR, file.filename)
        with open(backup_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Extract model name from filename (faiss_backup_MODEL_YYYYMMDDHHMM.zip)
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

@app.delete("/delete/{uid}/")
def delete_file(uid: str):
    """Deletes a specific uploaded file by UID."""
    metadata = load_metadata()

    if uid not in metadata:
        raise HTTPException(status_code=404, detail="File not found.")

    file_path = os.path.join(UPLOAD_DIR, metadata[uid]["stored_filename"])

    # Remove file if it exists
    if os.path.exists(file_path):
        os.remove(file_path)

    # Remove entry from metadata
    del metadata[uid]
    save_metadata(metadata)

    return {"message": f"File with UID {uid} deleted successfully."}

@app.delete("/delete/all/")
def delete_all_files():
    """Deletes all uploaded files, clears metadata, and removes FAISS indexes."""
    try:
        metadata = load_metadata()
        files_deleted = 0

        # **Delete each file (handling missing files gracefully)**
        for uid in list(metadata.keys()):
            file_path = os.path.join(UPLOAD_DIR, metadata[uid]["stored_filename"])
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                    files_deleted += 1
                    print(f"DEBUG: Deleted file {file_path}")
                except Exception as e:
                    print(f"ERROR: Failed to delete {file_path}: {str(e)}")
            else:
                print(f"WARNING: File {file_path} already missing. Removing from metadata.")

            # **Remove metadata entry**
            del metadata[uid]

        # **Save cleaned metadata**
        save_metadata(metadata)

        # **Ensure upload directory is fully cleared**
        shutil.rmtree(UPLOAD_DIR, ignore_errors=True)
        os.makedirs(UPLOAD_DIR, exist_ok=True)
        print("DEBUG: Upload directory reset successfully.")

        # **Remove all FAISS indexes dynamically**
        faiss_indexes_deleted = 0
        for folder in os.listdir(FAISS_BASE_PATH):
            if folder.startswith("faiss_index_"):
                faiss_path = os.path.join(FAISS_BASE_PATH, folder)
                try:
                    shutil.rmtree(faiss_path)
                    faiss_indexes_deleted += 1
                    print(f"DEBUG: Deleted FAISS index {faiss_path}")
                except Exception as e:
                    print(f"ERROR: Failed to delete FAISS index {faiss_path}: {str(e)}")

        return {
            "message": "All files, metadata, and FAISS indexes have been deleted.",
            "files_deleted": files_deleted,
            "faiss_indexes_deleted": faiss_indexes_deleted
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting files and indexes: {str(e)}")

@app.get("/query/")
def query_extracted_text(question: str):
    """Queries the extracted text using AI."""
    try:
        response = query_ai(question)
        return {"question": question, "answer": response}
    except Exception as e:
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
    """Switch to a new model and update the global FAISS path."""
    result = switch_model(new_model)

    # Ensure main.py knows the new model
    global OLLAMA_MODEL
    OLLAMA_MODEL = new_model
    os.environ["OLLAMA_MODEL"] = new_model
    os.environ["FAISS_INDEX_PATH"] = f"{FAISS_BASE_PATH}/faiss_index_{new_model}"

    return result


