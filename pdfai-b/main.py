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

@app.get("/current_model/")
def get_current_model():
    """Returns the current model being used for FAISS and queries."""
    return {"current_model": OLLAMA_MODEL}

@app.get("/faiss_model/")
def get_active_faiss_model():
    """Returns the FAISS index in use."""
    return {"active_faiss_index": os.getenv("FAISS_INDEX_PATH")}

@app.post("/switch_model/")
def switch_model(new_model: str):
    """Switches to the specified model, creates a new FAISS index, and deletes the old one."""
    global OLLAMA_MODEL

    print(f"DEBUG: Switching model to {new_model}...")

    # **Get current FAISS path before switching**
    old_faiss_path = os.getenv("FAISS_INDEX_PATH", f"{FAISS_BASE_PATH}/faiss_index_mistral")

    # **Switch to the new model**
    OLLAMA_MODEL = new_model
    os.environ["OLLAMA_MODEL"] = new_model

    # **Persist model selection**
    with open(MODEL_TRACK_FILE, "w") as f:
        f.write(new_model)

    # **Create new FAISS index path**
    new_faiss_path = f"{FAISS_BASE_PATH}/faiss_index_{new_model}"
    os.environ["FAISS_INDEX_PATH"] = new_faiss_path  

    # **Create new FAISS index**
    print(f"DEBUG: Creating new FAISS index for {new_model}...")
    os.makedirs(new_faiss_path, exist_ok=True)
    embeddings = OllamaEmbeddings(model=new_model, base_url="http://ollama:11434")
    FAISS.from_texts(["FAISS Initialized"], embeddings).save_local(new_faiss_path)

    # **Delete the old FAISS index since it's no longer used**
    if os.path.exists(old_faiss_path):
        print(f"DEBUG: Deleting old FAISS index: {old_faiss_path}")
        shutil.rmtree(old_faiss_path, ignore_errors=True)

    return {"message": f"Model switched to {new_model}. New FAISS index created, old index deleted."}

@app.delete("/delete/all/")
def delete_all_files():
    """Deletes all uploaded files and all FAISS indexes."""
    try:
        shutil.rmtree(UPLOAD_DIR, ignore_errors=True)
        os.makedirs(UPLOAD_DIR, exist_ok=True)

        for model_index in os.listdir(FAISS_BASE_PATH):
            if model_index.startswith("faiss_index_"):
                shutil.rmtree(f"{FAISS_BASE_PATH}/{model_index}", ignore_errors=True)

        if os.path.exists(MODEL_TRACK_FILE):
            os.remove(MODEL_TRACK_FILE)

        return {"message": "All files and FAISS indexes have been deleted."}
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
