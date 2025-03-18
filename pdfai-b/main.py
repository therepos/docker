import os
import shutil
import random
import string
import zipfile
import json
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
FAISS_BACKUP = "/app/faiss_backup.zip"
MODEL_TRACK_FILE = "/app/faiss_model.txt"

# Environment variables
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "mistral")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")

app = FastAPI()
os.makedirs(UPLOAD_DIR, exist_ok=True)

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

                uploaded_files.append({"file": file.filename, "message": "Text stored in FAISS."})
            else:
                uploaded_files.append({"file": file.filename, "message": "No text extracted."})

        except Exception as e:
            uploaded_files.append({"file": file.filename, "message": f"Error: {str(e)}"})

    return {"message": "Upload complete", "results": uploaded_files}

@app.get("/list_files/")
def list_files():
    """Lists all uploaded files."""
    return {"files": os.listdir(UPLOAD_DIR)}

@app.get("/download/{filename}/")
async def download_file(filename: str):
    """Downloads a specific uploaded file."""
    file_path = os.path.join(UPLOAD_DIR, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found.")
    return FileResponse(file_path, media_type="text/plain", filename=filename)

@app.get("/export_all/")
async def export_all_files():
    """Compresses all uploaded files into a ZIP and returns it."""
    zip_path = "/app/export/data_export.zip"
    shutil.make_archive("/app/export/data_export", 'zip', UPLOAD_DIR)
    return FileResponse(zip_path, media_type="application/zip", filename="data_export.zip")

@app.get("/export_faiss/")
async def export_faiss():
    """Exports FAISS index as a backup file."""
    try:
        save_model_label()
        shutil.make_archive("/app/faiss_backup", 'zip', FAISS_INDEX_PATH)
        return FileResponse(FAISS_BACKUP, media_type="application/zip", filename="faiss_backup.zip")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error exporting FAISS index: {str(e)}")

@app.post("/import_faiss/")
async def import_faiss(file: UploadFile = File(...)):
    """Restores FAISS index from a backup file."""
    try:
        with open(FAISS_BACKUP, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        shutil.unpack_archive(FAISS_BACKUP, FAISS_INDEX_PATH, "zip")

        imported_model = load_model_label()
        if imported_model and imported_model != OLLAMA_MODEL:
            switch_model(imported_model)

        return {"message": "FAISS index successfully restored."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error importing FAISS index: {str(e)}")

@app.delete("/delete/{filename}/")
def delete_file(filename: str):
    """Deletes a specific uploaded file."""
    file_path = os.path.join(UPLOAD_DIR, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found.")

    os.remove(file_path)
    return {"message": f"File {filename} deleted successfully."}

@app.post("/switch_model/")
def switch_model(new_model: str):
    """Switches the model and forces reindexing to prevent FAISS mixing."""
    global OLLAMA_MODEL
    OLLAMA_MODEL = new_model
    os.environ["OLLAMA_MODEL"] = new_model

    # Clear FAISS
    embeddings = OllamaEmbeddings(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL)
    vector_store = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
    vector_store.delete_all()
    vector_store.save_local(FAISS_INDEX_PATH)
    
    save_model_label()
    return {"message": f"Model switched to {new_model}. FAISS reindexed."}
