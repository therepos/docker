import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import os

def extract_text(file_path):
    """Extracts text from a PDF or image file."""
    file_ext = file_path.split(".")[-1].lower()

    if file_ext == "pdf":
        return extract_text_from_pdf(file_path)
    elif file_ext in ["jpg", "jpeg", "png"]:
        return extract_text_from_image(file_path)
    else:
        return ""

def extract_text_from_pdf(pdf_path):
    """Extracts text from a PDF file."""
    try:
        doc = fitz.open(pdf_path)
        return "\n".join([page.get_text("text") for page in doc])
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
        return ""

def extract_text_from_image(image_path):
    """Extracts text from an image file using OCR."""
    try:
        image = Image.open(image_path)
        return pytesseract.image_to_string(image)
    except Exception as e:
        print(f"Error extracting text from image: {e}")
        return ""
