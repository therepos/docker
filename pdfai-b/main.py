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

# Load Ollama service name from environment variables (set in docker-compose)
OLLAMA_SERVICE = os.getenv("OLLAMA_SERVICE", "ollama")

app = FastAPI()

UPLOAD_DIR = "data"
METADATA_FILE = os.path.join(UPLOAD_DIR, "files_metadata.json")
os.makedirs(UPLOAD_DIR, exist_ok=True)

def save_metadata(metadata):
    """Ensures metadata file exists and writes data to it."""
    try:
        os.makedirs(UPLOAD_DIR, exist_ok=True)
        existing_metadata = load_metadata()
        existing_metadata.update(metadata)

        with open(METADATA_FILE, "w") as f:
            json.dump(existing_metadata, f, indent=4)

    except Exception as e:
        logging.error(f"Error saving metadata: {e}")

def load_metadata():
    """Loads metadata from a JSON file."""
    if not os.path.exists(METADATA_FILE):
        return {}
    with open(METADATA_FILE, "r") as f:
        return json.load(f)

def generate_uid():
    """Generate a unique 12-character alphanumeric UID."""
    metadata = load_metadata()
    existing_uids = set(metadata.keys())

    while True:
        uid = ''.join(random.choices(string.ascii_letters + string.digits, k=12))
        print(f"DEBUG: Generated UID: {uid}")  # ✅ Log UID generation
        if uid not in existing_uids:
            return uid

@app.get("/")
def about():
    """Returns basic information about the API."""
    return {"message": "PDF-AI API is running", "version": "1.0"}

@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    """Uploads a file, extracts text, stores in FAISS, and assigns a unique UID."""
    metadata = load_metadata()
    uid = generate_uid()
    text_filename = f"{uid}.txt"
    text_path = os.path.join(UPLOAD_DIR, text_filename)

    print(f"DEBUG: Upload started for {file.filename}, UID: {uid}")
    print(f"DEBUG: Using Ollama model: {os.getenv('OLLAMA_MODEL')} at {os.getenv('OLLAMA_SERVICE')}")

    try:
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        extracted_text = extract_text(file_path)
        print(f"DEBUG: Extracted text length: {len(extracted_text)} characters")

        if extracted_text.strip():
            with open(text_path, "w", encoding="utf-8") as text_file:
                text_file.write(extracted_text)

            chunks = chunk_text(extracted_text)
            print(f"DEBUG: Chunks created: {len(chunks)}")

            store_in_faiss(chunks, os.getenv("OLLAMA_MODEL", "mistral"), uid)
            print(f"DEBUG: Stored in FAISS with UID: {uid}")

            os.remove(file_path)

            metadata[uid] = {
                "uid": uid,
                "original_filename": file.filename,
                "stored_filename": text_filename,
                "size_kb": round(len(extracted_text) / 1024, 2),
                "last_modified": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "file_type": "PDF" if file.filename.lower().endswith(".pdf") else "Image",
            }
            save_metadata(metadata)

            return {"uid": uid, "message": "Text extracted and stored in FAISS."}
        else:
            return {"file": file.filename, "message": "No text extracted."}

    except Exception as e:
        print(f"DEBUG: Upload failed - {str(e)}")
        return {"file": file.filename, "message": f"Error processing file: {str(e)}"}

@app.post("/upload/bulk/")
async def bulk_upload(files: list[UploadFile] = File(...)):
    """Uploads multiple files, extracts text, and stores them in FAISS."""
    uploaded_files = []
    for file in files:
        try:
            result = await upload_file(file)  # Call existing function
            uploaded_files.append(result)
        except Exception as e:
            uploaded_files.append({"file": file.filename, "error": str(e)})

    return {"message": "Bulk upload complete", "results": uploaded_files}

@app.get("/files/")
def list_extracted_files():
    """Lists available extracted text files."""
    metadata = load_metadata()
    if not metadata:
        return {"message": "No extracted text files found."}
    return {"extracted_files": list(metadata.values())}

@app.delete("/delete/")
def delete_extracted_text(uid: str):
    """Deletes an extracted text file, its embeddings, and removes its metadata."""
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

@app.delete("/delete/bulk/")
def bulk_delete(uids: list[str]):
    """Deletes multiple extracted files and embeddings."""
    metadata = load_metadata()
    deleted_files = []

    for uid in uids:
        if uid not in metadata:
            deleted_files.append({"uid": uid, "error": "File not found."})
            continue

        text_path = os.path.join(UPLOAD_DIR, metadata[uid]["stored_filename"])

        try:
            # Remove FAISS vector
            vector_store = FAISS.load_local("/app/faiss_index", OllamaEmbeddings(), allow_dangerous_deserialization=True)
            vector_store.delete([uid])
            vector_store.save_local("/app/faiss_index")  # Ensure FAISS index updates after deletion

            # Remove text file
            if os.path.exists(text_path):
                os.remove(text_path)

            # Remove metadata
            del metadata[uid]
            save_metadata(metadata)

            deleted_files.append({"uid": uid, "message": "Deleted successfully."})

        except Exception as e:
            deleted_files.append({"uid": uid, "error": str(e)})

    return {"message": "Bulk delete complete", "results": deleted_files}

@app.get("/query/")
def query_extracted_text(uid: str = None, question: str = None):
    """Queries extracted text for a specific UID or all documents."""
    if not question:
        return {"message": "Please provide a question to query."}

    try:
        response = query_ai(question, uid)
        return {"uid": uid, "question": question, "answer": response}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

@app.post("/switch_model/")
def switch_model(model: str):
    """Switch model for Ollama."""
    if model not in ['mistral', 'gemma3', 'llama3.1', 'llama3.2', 'llama3.3', 'phi4', 'deepseek-r1', 'qwq']:
        raise HTTPException(status_code=400, detail="Invalid model specified")

    os.environ["OLLAMA_MODEL"] = model
    return {"message": f"Model switched to {model}"}

@app.post("/reindex_embeddings/")
def reindex_embeddings():
    """Recompute and update all embeddings using the currently selected model."""
    model = os.getenv("OLLAMA_MODEL", "mistral")  # Get the active model
    embeddings = OllamaEmbeddings(model=model)

    # Load existing FAISS index
    faiss_index_path = "/app/faiss_index"
    if not os.path.exists(faiss_index_path):
        raise HTTPException(status_code=400, detail="FAISS index not found.")

    vector_store = FAISS.load_local(faiss_index_path, embeddings, allow_dangerous_deserialization=True)

    # Load metadata to retrieve all stored UIDs
    metadata = load_metadata()
    if not metadata:
        raise HTTPException(status_code=400, detail="No stored documents found.")

    all_texts = []
    all_metadata = []

    # Extract stored texts for re-embedding
    for uid, data in metadata.items():
        text_path = os.path.join("data", data["stored_filename"])
        if os.path.exists(text_path):
            with open(text_path, "r", encoding="utf-8") as file:
                text = file.read()
                all_texts.append(text)
                all_metadata.append({"uid": uid})

    if not all_texts:
        raise HTTPException(status_code=400, detail="No valid text data found.")

    # Generate new embeddings using the new model
    new_vector_store = FAISS.from_texts(all_texts, embeddings, metadatas=all_metadata)
    new_vector_store.save_local(faiss_index_path)

    return {"message": f"Re-indexed all documents using model {model}."}
