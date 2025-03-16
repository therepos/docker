from fastapi import FastAPI, UploadFile, File, HTTPException
from typing import List
import shutil
import os
import pytesseract
from PIL import Image
import fitz  # PyMuPDF
import logging
from extract import process_pdf  # Import process_pdf for PDF processing

app = FastAPI()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

UPLOAD_DIR = "data"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.get("/")
def read_root():
    return {"message": "PDF-AI API is running"}

# PDF Text Extraction
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text("text") + "\n"
    return text if text.strip() else None

# OCR for Image Files (JPEG/PNG)
def extract_text_from_image(image_path):
    img = Image.open(image_path)
    return pytesseract.image_to_string(img)

# API to Upload Multiple PDFs
@app.post("/upload_pdfs/")
async def upload_pdfs(files: List[UploadFile] = File(...)):
    """Uploads multiple PDF files and extracts text."""
    responses = []
    for file in files:
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        text_path = file_path.replace(".pdf", ".txt")  # Set path for extracted text

        logger.info(f"Received PDF upload: {file.filename}")
        try:
            # Save the uploaded file
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)

            # Extract text from the PDF
            extracted_text = process_pdf(file_path, text_path)  # Use existing process_pdf for PDFs

            if extracted_text:
                with open(text_path, "w", encoding="utf-8") as text_file:
                    text_file.write(extracted_text)
                logger.info(f"Text extracted and saved to {text_path}")
                os.remove(file_path)  # Optionally remove the original file after processing
                responses.append({"file": file.filename, "message": f"Text extracted to {text_path}"})
            else:
                logger.error(f"No text extracted from {file.filename}")
                responses.append({"file": file.filename, "message": "No text extracted."})
        except Exception as e:
            logger.error(f"Error processing PDF {file.filename}: {e}")
            responses.append({"file": file.filename, "message": "Internal Server Error"})
    
    return responses

# API to Upload Multiple Images (JPEG/PNG)
@app.post("/upload_images/")
async def upload_images(files: List[UploadFile] = File(...)):
    """Uploads multiple image files (JPEG/PNG) and extracts text."""
    responses = []
    for file in files:
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        text_path = file_path.replace(file.filename.split('.')[-1], "txt")  # Set path for extracted text

        logger.info(f"Received image upload: {file.filename}")
        try:
            # Save the uploaded image
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)

            # Extract text from the image
            extracted_text = extract_text_from_image(file_path)  # OCR for images
            logger.info(f"Text extracted from image: {file.filename}")

            if extracted_text:
                with open(text_path, "w", encoding="utf-8") as text_file:
                    text_file.write(extracted_text)
                logger.info(f"Text extracted and saved to {text_path}")
                os.remove(file_path)  # Optionally remove the original file after processing
                responses.append({"file": file.filename, "message": f"Text extracted to {text_path}"})
            else:
                logger.error(f"No text extracted from {file.filename}")
                responses.append({"file": file.filename, "message": "No text extracted."})
        except Exception as e:
            logger.error(f"Error processing image {file.filename}: {e}")
            responses.append({"file": file.filename, "message": "Internal Server Error"})
    
    return responses
