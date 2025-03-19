import os
import re
import shutil
import random
import string
import zipfile
import json
from datetime import datetime
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from extract import extract_text
from store import store_in_faiss
from process import chunk_text
from query import query_ai
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings

# Constants
UPLOAD_DIR = "data"
FAISS_INDEX_PATH = "/app/faiss_index"
EXPORT_DIR = "/app/export"
FAISS_BACKUP_TEMPLATE = "/app/export/faiss_backup_{}.zip"
MODEL_TRACK_FILE = "/app/faiss_model.txt"
METADATA_FILE = os.path.join(UPLOAD_DIR, "files_metadata.json")

# Environment variables
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "mistral")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")

app = FastAPI()
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(EXPORT_DIR, exist_ok=True)

def save_metadata(metadata):
    """Writes metadata to a JSON file."""
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

def save_model_label():
    """Saves the current model to a file for export labeling."""
    with open(MODEL_TRACK_FILE, "w") as f:
        f.write(OLLAMA_MODEL)

def load_model_label():
    """Loads the model label from an exported FAISS index."""
    if os.path.exists(MODEL_TRACK_FILE):
        with open(MODEL_TRACK_FILE, "r") as f:
            return f.read().strip()
    return None

@app.get("/")
def about():
    """Returns API status."""
    return {"message": "PDF-AI API is running", "version": "1.0", "model": OLLAMA_MODEL}

@app.post("/upload/")
async def upload_file(files: list[UploadFile] = File(...)):
    """Uploads files, extracts text if necessary, and stores embeddings in FAISS."""
    metadata = load_metadata()
    uploaded_files = []
    save_model_label()

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
    metadata = load_metadata()
    return {"files": metadata}

@app.get("/download/{uid}/")
async def download_file(uid: str):
    """Downloads extracted text file by UID."""
    metadata = load_metadata()

    if uid not in metadata:
        raise HTTPException(status_code=404, detail="File not found.")

    file_path = os.path.join(UPLOAD_DIR, metadata[uid]["stored_filename"])

    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Extracted text file not found.")

    return FileResponse(file_path, media_type="text/plain", filename=metadata[uid]["original_filename"] + ".txt")

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

@app.get("/query/")
def query_extracted_text(question: str):
    """Queries the extracted text using AI."""
    try:
        response = query_ai(question)
        return {"question": question, "answer": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

@app.delete("/delete/{filename}/")
def delete_file(filename: str):
    """Deletes a specific uploaded file."""
    metadata = load_metadata()
    file_path = os.path.join(UPLOAD_DIR, filename)

    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found.")

    os.remove(file_path)
    for uid, data in list(metadata.items()):
        if data["stored_filename"] == filename:
            del metadata[uid]

    save_metadata(metadata)
    return {"message": f"File {filename} deleted successfully."}

@app.delete("/delete/all/")
def delete_all_files():
    """Deletes all uploaded files, clears extracted text, and resets FAISS."""
    try:
        # Step 1: Remove all data files
        shutil.rmtree(UPLOAD_DIR, ignore_errors=True)
        os.makedirs(UPLOAD_DIR, exist_ok=True)

        # Step 2: Reset FAISS
        embeddings = OllamaEmbeddings(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL)
        vector_store = FAISS.from_texts(["FAISS Initialized"], embeddings)
        vector_store.save_local(FAISS_INDEX_PATH)

        # Clear metadata
        if os.path.exists(METADATA_FILE):
            os.remove(METADATA_FILE)

        return {"message": "All files, extracted text, and FAISS index have been reset."}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error resetting system: {str(e)}")

@app.post("/switch_model/")
def switch_model(new_model: str):
    """Switches the model and reprocesses FAISS to prevent mixed embedding dimensions."""
    global OLLAMA_MODEL
    OLLAMA_MODEL = new_model
    os.environ["OLLAMA_MODEL"] = new_model

    metadata = load_metadata()
    all_texts = []

    # Extract all stored text files
    for uid, data in metadata.items():
        text_path = os.path.join(UPLOAD_DIR, data["stored_filename"])
        if os.path.exists(text_path):
            with open(text_path, "r", encoding="utf-8") as f:
                text = f.read().strip()
                if text:
                    all_texts.append(text)

    # **Clear FAISS before reprocessing** to avoid mixed dimensions
    shutil.rmtree(FAISS_INDEX_PATH, ignore_errors=True)
    os.makedirs(FAISS_INDEX_PATH, exist_ok=True)

    # Reprocess FAISS with the new model
    if all_texts:
        embeddings = OllamaEmbeddings(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL)
        vector_store = FAISS.from_texts(all_texts, embeddings)
        vector_store.save_local(FAISS_INDEX_PATH)

    save_model_label()
    return {"message": f"Model switched to {new_model}. FAISS fully reprocessed with the new model."}

@app.get("/current_model/")
def get_current_model():
    """Returns the current model being used for FAISS and queries."""
    return {"current_model": OLLAMA_MODEL}


