import os
import logging
import datetime
import json
import shutil
import random
import string
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from extract import process_pdf
from store import store_text_in_faiss
from process import chunk_text
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

@app.get("/")
def about():
    """Returns basic information about the API."""
    return {"message": "PDF-AI API is running", "version": "1.0", "features": ["Upload PDF", "Query AI", "Download Extracted Text", "Delete Files"]}

@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    """Uploads a PDF, extracts text, stores in FAISS, and assigns a short UID."""
    metadata = load_metadata()
    
    uid = ''.join(random.choices(string.ascii_letters + string.digits, k=12))  # Generate a 12-character UID
    text_filename = f"{uid}.txt"
    text_path = os.path.join(UPLOAD_DIR, text_filename)

    logger = logging.getLogger(__name__)
    logger.info(f"Received file upload: {file.filename}")

    try:
        pdf_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(pdf_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Extract text using extract.py
        process_pdf(pdf_path, text_path)

        # Read extracted text
        with open(text_path, "r", encoding="utf-8") as text_file:
            extracted_text = text_file.read()

        if extracted_text.strip():
            # Chunk and store in FAISS using store.py
            chunks = chunk_text(extracted_text)
            store_text_in_faiss(chunks)

            os.remove(pdf_path)  # Delete original PDF

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

@app.get("/query/")
def query_extracted_text(question: str):
    """Queries the extracted text using AI (Ollama or OpenAI)."""
    try:
        response = query_ai(question)
        return {"question": question, "answer": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")
