import os
import shutil
import random
import string
import zipfile
import json
import datetime
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
                    "last_modified": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
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
    zip_path = os.path.join(EXPORT_DIR, "data_export.zip")
    shutil.make_archive(os.path.join(EXPORT_DIR, "data_export"), 'zip', UPLOAD_DIR)
    return FileResponse(zip_path, media_type="application/zip", filename="data_export.zip")

@app.get("/export_faiss/")
async def export_faiss():
    """Exports FAISS index as a backup file including the model name."""
    try:
        save_model_label()
        model_name = OLLAMA_MODEL.replace("/", "_")
        backup_path = FAISS_BACKUP_TEMPLATE.format(model_name)
        shutil.make_archive(backup_path.replace(".zip", ""), 'zip', FAISS_INDEX_PATH)
        return FileResponse(backup_path, media_type="application/zip", filename=f"faiss_backup_{model_name}.zip")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error exporting FAISS index: {str(e)}")

@app.post("/import_faiss/")
async def import_faiss(file: UploadFile = File(...)):
    """Restores FAISS index from a backup file and ensures model consistency."""
    try:
        backup_path = os.path.join(EXPORT_DIR, file.filename)
        with open(backup_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        shutil.unpack_archive(backup_path, FAISS_INDEX_PATH, "zip")

        imported_model = load_model_label()
        if imported_model and imported_model != OLLAMA_MODEL:
            switch_model(imported_model)

        return {"message": "FAISS index successfully restored."}
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
    """Switches the model and forces reindexing to prevent FAISS mixing."""
    global OLLAMA_MODEL
    OLLAMA_MODEL = new_model
    os.environ["OLLAMA_MODEL"] = new_model

    # Reinitialize FAISS instead of delete_all()
    embeddings = OllamaEmbeddings(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL)
    vector_store = FAISS.from_texts(["FAISS Initialized"], embeddings)
    vector_store.save_local(FAISS_INDEX_PATH)

    save_model_label()
    return {"message": f"Model switched to {new_model}. FAISS reindexed."}

