import fitz  # PyMuPDF
import pytesseract
from pdf2image import convert_from_path
from PIL import Image
from ebooklib import epub
from mobi import extract
import os

def extract_text_from_epub(epub_path):
    """Extract text from an EPUB file."""
    book = epub.read_epub(epub_path)
    text = ""
    for item in book.get_items():
        if item.get_type() == ebooklib.ITEM_DOCUMENT:
            text += item.get_body().decode("utf-8")
    return text

def extract_text_from_mobi(mobi_path):
    """Extract text from a MOBI file."""
    return extract(mobi_path)

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

def extract_text_from_image(image_path):
    """Extract text from an image (JPEG/PNG) using OCR."""
    img = Image.open(image_path)
    return pytesseract.image_to_string(img)

def extract_text(file_path):
    """Extract text based on the file type (PDF, scanned PDF, epub, mobi, or Image)."""
    if file_path.lower().endswith('.epub'):
        return extract_text_from_epub(file_path)
    elif file_path.lower().endswith('.mobi'):
        return extract_text_from_mobi(file_path)
    elif file_path.lower().endswith('.pdf'):
        # If it's a PDF, try to extract text using PyMuPDF
        text = extract_text_from_pdf(file_path)
        if not text:  # If no text extracted, try OCR on scanned PDFs
            text = extract_text_from_scanned_pdf(file_path)
        return text
    elif file_path.lower().endswith(('.jpg', '.jpeg', '.png')):
        # If it's an image, use OCR for text extraction
        return extract_text_from_image(file_path)
    else:
        raise ValueError("Unsupported file type")
