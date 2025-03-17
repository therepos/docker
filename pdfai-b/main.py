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
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "mistral")    

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
            store_in_faiss(chunks, OLLAMA_MODEL, uid)

            os.remove(file_path)  # Delete original file

            metadata[uid] = {
                "uid": uid,
                "original_filename": file.filename,
                "stored_filename": text_filename,
                "size_kb": round(len(extracted_text) / 1024, 2),
                "last_modified": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "file_type": "PDF" if file.filename.lower().endswith(".pdf") else "Image",
                "model": OLLAMA_MODEL  # Store the model used for FAISS
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
    files_info = []
    for uid, file_info in metadata.items():
        # Fetch "model" key from metadata
        model_used = file_info.get("model", "Unknown")
        files_info.append({
            "uid": uid,
            "filename": file_info["original_filename"],
            "size_kb": file_info["size_kb"],
            "last_modified": file_info["last_modified"],
            "model_used": model_used  # Show the model used
        })
    return {"extracted_files": files_info}

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
    """Deletes an extracted text file, its embeddings, and removes its metadata."""
    metadata = load_metadata()

    if uid not in metadata:
        raise HTTPException(status_code=404, detail="File not found.")

    text_path = os.path.join(UPLOAD_DIR, metadata[uid]["stored_filename"])

    # Remove the text from FAISS as well
    vector_store = FAISS.load_local("/app/faiss_index", OllamaEmbeddings(), allow_dangerous_deserialization=True)
    vector_store.delete([uid])  # Delete the corresponding embedding

    try:
        if os.path.exists(text_path):
            os.remove(text_path)

        del metadata[uid]
        save_metadata(metadata)

        return {"uid": uid, "message": "File and embedding deleted successfully."}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting file: {str(e)}")

@app.get("/query/")
def query_extracted_text(uid: str = None, question: str = None):
    """Queries the extracted text for a specific UID, or all documents if UID is not specified."""
 
    # Handle case where no question is provided
    if not question:
        return {"message": "Please provide a question to query."}

    metadata = load_metadata()

    try:
        # If UID is provided, filter by it
        if uid:
            if uid not in metadata:
                raise HTTPException(status_code=404, detail="UID not found.")
            # Pass UID along to query_ai for specific document query
            response = query_ai(question, uid)
        else:
            # If no UID is provided, run query on all documents
            response = query_ai(question)

        return {"uid": uid, "question": question, "answer": response}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

@app.post("/switch_model/")
def switch_model(model: str):
    """Switch model for Ollama."""
    # Dynamic switching within the current application session
    if model not in ['mistral', 'llama3.2', 'gemma3']:
        raise HTTPException(status_code=400, detail="Invalid model specified")

    # Update the environment variable for OLLAMA_MODEL
    os.environ["OLLAMA_MODEL"] = model

    return {"message": f"Model switched to {model}"}