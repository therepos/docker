import random
import string
import json
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.responses import FileResponse
import os
import shutil
import logging
import datetime
from process import chunk_text
from store import store_in_faiss
from query import query_ai

app = FastAPI()

UPLOAD_DIR = "data"
METADATA_FILE = os.path.join(UPLOAD_DIR, "files_metadata.json")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Load existing metadata or create a new one
if not os.path.exists(METADATA_FILE):
    with open(METADATA_FILE, "w") as f:
        json.dump({}, f)

def save_metadata(metadata):
    with open(METADATA_FILE, "w") as f:
        json.dump(metadata, f, indent=4)

def load_metadata():
    with open(METADATA_FILE, "r") as f:
        return json.load(f)

def generate_uid():
    """Generate a 12-character alphanumeric UID."""
    return ''.join(random.choices(string.ascii_letters + string.digits, k=12))

@app.get("/")
def about():
    """Returns basic information about the API."""
    return {"message": "PDF-AI API is running", "version": "1.0", "features": ["Upload PDF", "Query AI", "Download Extracted Text", "Delete Files"]}

@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    """Uploads a PDF, extracts text, stores in FAISS, and assigns a short UID."""
    metadata = load_metadata()
    
    uid = generate_uid()  # Generate a 12-character UID
    text_filename = f"{uid}.txt"
    file_path = os.path.join(UPLOAD_DIR, text_filename)

    logger = logging.getLogger(__name__)
    logger.info(f"Received file upload: {file.filename}")

    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        import fitz  # PyMuPDF
        doc = fitz.open(file_path)
        extracted_text = "\n".join([page.get_text("text") for page in doc])

        if extracted_text.strip():
            with open(file_path, "w", encoding="utf-8") as text_file:
                text_file.write(extracted_text)

            # Chunk and store in FAISS
            chunks = chunk_text(extracted_text)
            store_in_faiss(chunks)

            os.remove(file.filename)  # Delete original PDF

            # Save metadata
            metadata[uid] = {
                "uid": uid,
                "original_filename": file.filename,
                "stored_filename": text_filename,
                "size_kb": round(len(extracted_text) / 1024, 2),
                "last_modified": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            save_metadata(metadata)

            return {"uid": uid, "file": file.filename, "message": "Text extracted and stored in FAISS."}
        else:
            return {"file": file.filename, "message": "No text extracted."}

    except Exception as e:
        logger.error(f"Error processing file {file.filename}: {e}")
        return {"file": file.filename, "message": "Internal Server Error"}

@app.get("/files/")
def list_extracted_files():
    """Lists available extracted text files with UID, filename, size, and modified date."""
    metadata = load_metadata()
    if not metadata:
        return {"message": "No extracted text files found."}
    return {"extracted_files": list(metadata.values())}

@app.get("/download/")
async def download_extracted_text(uid: str):
    """Allows users to download extracted text using UID."""
    metadata = load_metadata()
    
    if uid not in metadata:
        raise HTTPException(status_code=404, detail="File not found.")

    text_path = os.path.join(UPLOAD_DIR, metadata[uid]["stored_filename"])

    if not os.path.exists(text_path):
        raise HTTPException(status_code=404, detail="Extracted text file not found.")

    return FileResponse(text_path, media_type="text/plain", filename=metadata[uid]["original_filename"] + ".txt")

@app.delete("/delete/")
def delete_extracted_text(uid: str):
    """Deletes an extracted text file and removes its metadata."""
    metadata = load_metadata()

    if uid not in metadata:
        raise HTTPException(status_code=404, detail="File not found.")

    text_path = os.path.join(UPLOAD_DIR, metadata[uid]["stored_filename"])

    try:
        if os.path.exists(text_path):
            os.remove(text_path)

        del metadata[uid]
        save_metadata(metadata)

        return {"uid": uid, "message": "File deleted successfully."}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting file: {str(e)}")
