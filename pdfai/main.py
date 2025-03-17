import os
import logging
import datetime
import json
import shutil
import random
import string
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from extract import extract_text
from store import store_in_faiss
from process import chunk_text
from query import query_ai

app = FastAPI()

UPLOAD_DIR = "data"
METADATA_FILE = os.path.join(UPLOAD_DIR, "files_metadata.json")
os.makedirs(UPLOAD_DIR, exist_ok=True)

def save_metadata(metadata):
    """Ensures metadata file exists and writes data to it."""
    try:
        os.makedirs(UPLOAD_DIR, exist_ok=True)  # Ensure directory exists

        # Load existing metadata (if any)
        if os.path.exists(METADATA_FILE):
            with open(METADATA_FILE, "r") as f:
                try:
                    existing_metadata = json.load(f)
                except json.JSONDecodeError:
                    existing_metadata = {}
        else:
            existing_metadata = {}

        # Merge new metadata with existing metadata
        existing_metadata.update(metadata)

        # Write updated metadata
        with open(METADATA_FILE, "w") as f:
            json.dump(existing_metadata, f, indent=4)

    except Exception as e:
        logging.error(f"Error saving metadata: {e}")

def load_metadata():
    if not os.path.exists(METADATA_FILE):
        return {}  # Return an empty dictionary if the file doesn't exist
    with open(METADATA_FILE, "r") as f:
        return json.load(f)

def generate_uid():
    """Generate a unique 12-character alphanumeric UID."""
    metadata = load_metadata()
    existing_uids = set(metadata.keys())  # Get all existing UIDs

    while True:
        uid = ''.join(random.choices(string.ascii_letters + string.digits, k=12))
        if uid not in existing_uids:  # Ensure uniqueness
            return uid

@app.get("/")
def about():
    """Returns basic information about the API."""
    return {"message": "PDF-AI API is running", "version": "1.0", "features": ["Upload PDF/Image", "Query AI", "Download Extracted Text", "Delete Files"]}

@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    """Uploads a PDF or image, extracts text, stores in FAISS, and assigns a unique UID."""
    metadata = load_metadata()
    
    uid = generate_uid()
    text_filename = f"{uid}.txt"
    text_path = os.path.join(UPLOAD_DIR, text_filename)

    logger = logging.getLogger(__name__)
    logger.info(f"Received file upload: {file.filename}")

    try:
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Extract text using extract.py
        extracted_text = extract_text(file_path)

        if extracted_text.strip():
            with open(text_path, "w", encoding="utf-8") as text_file:
                text_file.write(extracted_text)

            # Chunk and store in FAISS
            chunks = chunk_text(extracted_text)
            store_in_faiss(chunks)

            os.remove(file_path)  # Delete original file

            metadata[uid] = {
                "uid": uid,
                "original_filename": file.filename,
                "stored_filename": text_filename,
                "size_kb": round(len(extracted_text) / 1024, 2),
                "last_modified": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "file_type": "PDF" if file.filename.lower().endswith(".pdf") else "Image"
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

@app.get("/query/")
def query_extracted_text(question: str):
    """Queries the extracted text using AI (Ollama or OpenAI)."""
    try:
        response = query_ai(question)
        return {"question": question, "answer": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")
