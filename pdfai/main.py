import os
import re
import shutil
import random
import string
import zipfile
import json
import traceback
import time
import threading
import smtplib
import logging
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import FileResponse, JSONResponse
from extract import extract_text
from store import store_in_faiss, initialize_faiss
from process import chunk_text
from query import query_ai
from indexer import switch_model
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

UPLOAD_DIR = "data"
EXPORT_DIR = "/app/export"
FAISS_BASE_PATH = "/app"
MODEL_TRACK_FILE = "/app/active_model.txt"
METADATA_FILE = os.path.join(UPLOAD_DIR, "files_metadata.json")

if os.path.exists(MODEL_TRACK_FILE):
    with open(MODEL_TRACK_FILE, "r") as f:
        OLLAMA_MODEL = f.read().strip()
else:
    OLLAMA_MODEL = "mistral"

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")
FAISS_INDEX_PATH = f"{FAISS_BASE_PATH}/faiss_index_{OLLAMA_MODEL}"
os.environ.update({
    "OLLAMA_BASE_URL": OLLAMA_BASE_URL,
    "OLLAMA_MODEL": OLLAMA_MODEL,
    "FAISS_INDEX_PATH": FAISS_INDEX_PATH
})

logger.info(f"Loaded OLLAMA_BASE_URL: {OLLAMA_BASE_URL}")
logger.info(f"Loaded OLLAMA_MODEL: {OLLAMA_MODEL}")
logger.info(f"FAISS Index Path: {FAISS_INDEX_PATH}")

initialize_faiss()

app = FastAPI()
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(EXPORT_DIR, exist_ok=True)

# === Utility Functions ===
def save_metadata(metadata):
    backup_path = METADATA_FILE + ".bak"
    if os.path.exists(METADATA_FILE):
        shutil.copy(METADATA_FILE, backup_path)
    with open(METADATA_FILE, "w") as f:
        json.dump(metadata, f, indent=4)

def load_metadata():
    if not os.path.exists(METADATA_FILE):
        return {}
    with open(METADATA_FILE, "r") as f:
        return json.load(f)

def generate_uid():
    return ''.join(random.choices(string.ascii_letters + string.digits, k=12))

# === API Endpoints ===
@app.get("/")
def about():
    return {"message": "PDF-AI API is running", "version": "1.0", "model": OLLAMA_MODEL}

@app.get("/list_files/")
def list_files():
    return {"files": load_metadata()}

@app.get("/current_model/")
def get_current_model():
    return {"current_model": OLLAMA_MODEL}

@app.get("/faiss_model/")
def get_active_faiss_model():
    return {"active_faiss_index": os.getenv("FAISS_INDEX_PATH")}

@app.post("/switch_model/")
def switch_model_endpoint(new_model: str):
    global OLLAMA_MODEL, FAISS_INDEX_PATH
    result = switch_model(new_model)
    OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", new_model)
    FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH", f"{FAISS_BASE_PATH}/faiss_index_{OLLAMA_MODEL}")
    logger.info(f"Switched model: {OLLAMA_MODEL}")
    return result

@app.get("/query/")
def query_extracted_text(question: str, uid: str = None):
    try:
        logger.info(f"Received query: {question}")
        embeddings = OllamaEmbeddings(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL)
        vector_store = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
        retriever = vector_store.as_retriever()

        if uid:
            retriever.search_kwargs = {"filter": {"source": uid}}

        docs = retriever.get_relevant_documents(question)
        if all("FAISS_INIT" in doc.page_content for doc in docs):
            return {"question": question, "file": uid or "all", "answer": "No file found."}

        response = query_ai(question)
        return {"question": question, "file": uid or "all", "answer": response}

    except Exception as e:
        logger.error(f"Query failed: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

@app.get("/export_all/")
async def export_all_files():
    zip_path = os.path.join(EXPORT_DIR, "data_export.zip")
    shutil.make_archive(os.path.join(EXPORT_DIR, "data_export"), 'zip', UPLOAD_DIR)
    return FileResponse(zip_path, media_type="application/zip", filename="data_export.zip")

@app.get("/export_faiss/")
async def export_faiss():
    try:
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
    try:
        backup_path = os.path.join(EXPORT_DIR, file.filename)
        with open(backup_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        match = re.search(r"faiss_backup_(.*?)_\d{12}\.zip", file.filename)
        if not match:
            raise HTTPException(status_code=400, detail="Invalid FAISS backup filename format.")

        imported_model = match.group(1)
        global OLLAMA_MODEL, FAISS_INDEX_PATH

        if imported_model != OLLAMA_MODEL:
            logger.info(f"Switching model to match FAISS backup: {imported_model}")
            switch_model(imported_model)
            OLLAMA_MODEL = imported_model
            FAISS_INDEX_PATH = f"{FAISS_BASE_PATH}/faiss_index_{OLLAMA_MODEL}"
            os.environ["OLLAMA_MODEL"] = OLLAMA_MODEL
            os.environ["FAISS_INDEX_PATH"] = FAISS_INDEX_PATH

        if os.path.exists(FAISS_INDEX_PATH):
            shutil.rmtree(FAISS_INDEX_PATH, ignore_errors=True)
            os.makedirs(FAISS_INDEX_PATH, exist_ok=True)

        shutil.unpack_archive(backup_path, FAISS_INDEX_PATH, "zip")
        return {"message": f"FAISS index for model '{imported_model}' restored and activated."}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error importing FAISS index: {str(e)}")

@app.delete("/delete/all/")
def delete_all_files():
    try:
        global OLLAMA_MODEL
        if os.path.exists(UPLOAD_DIR):
            shutil.rmtree(UPLOAD_DIR, ignore_errors=True)
            os.makedirs(UPLOAD_DIR, exist_ok=True)

        if os.path.exists(METADATA_FILE):
            os.remove(METADATA_FILE)

        faiss_path = f"/app/faiss_index_{OLLAMA_MODEL}"
        if os.path.exists(faiss_path):
            shutil.rmtree(faiss_path, ignore_errors=True)
            os.makedirs(faiss_path, exist_ok=True)

        initialize_faiss()
        return {"message": "All files, metadata, and FAISS index have been reset."}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error resetting FAISS: {str(e)}")

@app.post("/extract_and_email/")
async def extract_only(file: UploadFile = File(...), email: str = Form(...)):
    uid = generate_uid()
    file_bytes = await file.read()
    threading.Thread(target=process_and_email, args=(file.filename, file_bytes, email, uid)).start()
    return {"status": "processing", "uid": uid}

def process_and_email(filename, file_bytes, email, uid):
    try:
        raw_path = os.path.join(UPLOAD_DIR, filename)
        text_path = os.path.join(EXPORT_DIR, f"{uid}.txt")
        with open(raw_path, "wb") as f:
            f.write(file_bytes)

        if filename.endswith(".txt"):
            content = file_bytes.decode("utf-8")
            cleaned = clean_text(content)
        else:
            raw = extract_text(raw_path)
            cleaned = clean_text(raw)

        with open(text_path, "w", encoding="utf-8") as f:
            f.write(cleaned)

        link = f"https://pdfai.threeminuteslab.com/export/{uid}"
        send_download_email(email, link)

    except Exception as e:
        logger.error(f"Background task failed: {e}")

@app.get("/export/{uid}")
async def export_txt(uid: str):
    path = os.path.join(EXPORT_DIR, f"{uid}.txt")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(path, media_type="text/plain", filename=f"{uid}.txt")

@app.post("/extract_and_download/")
async def extract_only(file: UploadFile = File(...)):
    uid = generate_uid()
    raw_path = os.path.join(UPLOAD_DIR, file.filename)
    text_filename = f"{uid}.txt"
    text_path = os.path.join(UPLOAD_DIR, text_filename)

    try:
        with open(raw_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        if file.filename.endswith(".txt"):
            with open(raw_path, "r", encoding="utf-8") as f:
                content = f.read()
            cleaned = clean_text(content)
        else:
            raw = extract_text(raw_path)
            cleaned = clean_text(raw)

        with open(text_path, "w", encoding="utf-8") as f:
            f.write(cleaned)

        return FileResponse(path=text_path, media_type="text/plain", filename=text_filename)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Extraction failed: {str(e)}")

def clean_text(text: str) -> str:
    text = re.sub(r"\r\n|\r", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    lines = [line.lstrip() for line in text.split("\n")]
    text = "\n".join(lines)
    text = merge_broken_lines(text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def merge_broken_lines(text: str) -> str:
    lines = text.split('\n')
    merged = []
    for i in range(len(lines)):
        current = lines[i].rstrip()
        if i < len(lines) - 1:
            next_line = lines[i + 1].lstrip()
            if current and not re.search(r"[.?!]['\"]?$", current) and re.match(r"^[a-z\"'\(\[{.,;!?]", next_line):
                lines[i + 1] = current + " " + next_line
                continue
        merged.append(current)
    return '\n'.join(merged)

def send_download_email(to_email: str, download_url: str):
    smtp_host = os.environ.get("SMTP_HOST", "smtp.sendgrid.net")
    smtp_port = int(os.environ.get("SMTP_PORT", "587"))
    sender_email = os.environ.get("SENDER_EMAIL")
    sender_user = os.environ.get("SMTP_USER", "apikey")
    sender_pass = os.environ.get("SMTP_PASS")

    if not sender_email or not sender_pass:
        logger.error("Missing SENDER_EMAIL or SMTP_PASS")
        return

    try:
        msg = MIMEText(f"Your file is ready.\n\nClick to download:\n{download_url}", "plain")
        msg["Subject"] = "Your extracted file is ready"
        msg["From"] = f"PDF AI <{sender_email}>"
        msg["To"] = to_email
        msg.add_header("X-SMTPAPI", '{"filters": {"clicktrack": {"settings": {"enable": 0}}}}')

        with smtplib.SMTP(smtp_host, smtp_port) as server:
            server.set_debuglevel(1)
            server.starttls()
            server.login(sender_user, sender_pass)
            server.sendmail(sender_email, to_email, msg.as_string())
        logger.info(f"Email sent to {to_email}")
    except Exception as e:
        logger.error(f"Failed to send email: {e}")
