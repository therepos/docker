import os
import shutil
import random
import string
import zipfile
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
FAISS_BACKUP = "/app/export/faiss_backup.zip"
MODEL_TRACK_FILE = "/app/export/faiss_model.txt"

app = FastAPI()
os.makedirs(UPLOAD_DIR, exist_ok=True)

def generate_uid():
    """Generates a unique 12-character alphanumeric UID."""
    return ''.join(random.choices(string.ascii_letters + string.digits, k=12))

def get_active_model():
    """Retrieves the currently used embedding model."""
    return os.getenv("OLLAMA_MODEL", "mistral")

def save_model_label():
    """Saves the current model to a file for export labeling."""
    with open(MODEL_TRACK_FILE, "w") as f:
        f.write(get_active_model())

def load_model_label():
    """Loads the model label from an exported FAISS index."""
    if os.path.exists(MODEL_TRACK_FILE):
        with open(MODEL_TRACK_FILE, "r") as f:
            return f.read().strip()
    return None

@app.get("/")
def about():
    """Returns API status."""
    return {"message": "PDF-AI API is running", "version": "1.0", "model": get_active_model()}

@app.post("/upload/")
async def upload_file(files: list[UploadFile] = File(...)):
    """Uploads files, extracts text if necessary, and stores embeddings in FAISS."""
    uploaded_files = []
    save_model_label()  # Ensure FAISS is labeled with the current model

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

@app.get("/export_faiss/")
async def export_faiss():
    """Exports FAISS index as a backup file, including model information."""
    try:
        save_model_label()  # Store the model label before export
        shutil.make_archive("/app/faiss_backup", 'zip', FAISS_INDEX_PATH)
        return FileResponse(FAISS_BACKUP, media_type="application/zip", filename="faiss_backup.zip")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error exporting FAISS index: {str(e)}")

@app.post("/import_faiss/")
async def import_faiss(file: UploadFile = File(...)):
    """Restores FAISS index from a backup file and forces reindexing if model mismatches."""
    try:
        with open(FAISS_BACKUP, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        shutil.unpack_archive(FAISS_BACKUP, FAISS_INDEX_PATH, "zip")

        imported_model = load_model_label()
        if imported_model and imported_model != get_active_model():
            switch_model(imported_model)

        return {"message": "FAISS index successfully restored."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error importing FAISS index: {str(e)}")

@app.post("/switch_model/")
def switch_model(new_model: str):
    """Switches the model and forces reindexing to prevent FAISS mixing."""
    os.environ["OLLAMA_MODEL"] = new_model
    clear_faiss()
    return {"message": f"Model switched to {new_model}. FAISS reindexed."}

@app.delete("/delete/all/")
def delete_all_files():
    """Deletes all uploaded files and clears FAISS index."""
    try:
        shutil.rmtree(UPLOAD_DIR, ignore_errors=True)
        os.makedirs(UPLOAD_DIR, exist_ok=True)
        clear_faiss()
        return {"message": "All files and embeddings have been deleted."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting files and embeddings: {str(e)}")

def clear_faiss():
    """Clears FAISS storage before reindexing or model switching."""
    embeddings = OllamaEmbeddings(model=get_active_model())
    vector_store = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
    vector_store.delete_all()
    vector_store.save_local(FAISS_INDEX_PATH)
    save_model_label()  # Update model label after clearing FAISS
