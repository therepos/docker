import fitz
import pytesseract
from pdf2image import convert_from_path
import os

def extract_text_from_pdf(pdf_path):
    """Extract text from a digital/text-based PDF."""
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text("text") + "\n"
    return text if text.strip() else None

def extract_text_from_scanned_pdf(pdf_path):
    """Extract text from a scanned/image-based PDF using OCR."""
    images = convert_from_path(pdf_path)
    text = ""
    for img in images:
        text += pytesseract.image_to_string(img) + "\n"
    return text

def process_pdf(pdf_path, output_path):
    """Extract text from PDFs and save it to a file."""
    text = extract_text_from_pdf(pdf_path) or extract_text_from_scanned_pdf(pdf_path)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(text)
    return output_path
