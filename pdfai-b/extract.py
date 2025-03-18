import os
import mobi
import html2text
import fitz  # PyMuPDF
import pytesseract
from pdf2image import convert_from_path
from PIL import Image
from ebooklib import epub

def extract_text_from_epub(epub_path):
    """Extract text from an EPUB file."""
    try:
        book = epub.read_epub(epub_path)
        text = ""
        for item in book.get_items():
            if item.get_type() == epub.ITEM_DOCUMENT:
                text += item.get_body().decode("utf-8")
        return text.strip() if text else "Error: No readable text in EPUB."
    except Exception as e:
        return f"Error extracting EPUB text: {str(e)}"

def extract_text_from_mobi(mobi_path):
    """Convert MOBI to text using the best available method."""
    try:
        # Convert MOBI to EPUB first (preferred)
        output_dir = os.path.dirname(mobi_path)
        mobi.extract(mobi_path, output_dir)

        # Find extracted EPUB file
        epub_files = [file for file in os.listdir(output_dir) if file.endswith(".epub")]
        if epub_files:
            epub_path = os.path.join(output_dir, epub_files[0])
            return extract_text_from_epub(epub_path)

        # Fallback: Extract MOBI to HTML and convert using html2text
        tempdir, filepath = mobi.extract(mobi_path)
        with open(filepath, "r", encoding="utf-8") as file:
            content = file.read()
            return html2text.html2text(content).strip() if content else "Error: No readable text in MOBI."

    except Exception as e:
        return f"Error extracting MOBI text: {str(e)}"

def extract_text_from_pdf(pdf_path):
    """Extract text from a digital/text-based PDF."""
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text("text") + "\n"
        return text.strip() if text.strip() else None
    except Exception as e:
        return f"Error extracting PDF text: {str(e)}"

def extract_text_from_scanned_pdf(pdf_path):
    """Extract text from a scanned/image-based PDF using OCR."""
    try:
        images = convert_from_path(pdf_path)
        text = ""
        for img in images:
            text += pytesseract.image_to_string(img) + "\n"
        return text.strip() if text else "Error: No readable text from OCR."
    except Exception as e:
        return f"Error extracting scanned PDF text: {str(e)}"

def extract_text_from_image(image_path):
    """Extract text from an image (JPEG/PNG) using OCR."""
    try:
        img = Image.open(image_path)
        return pytesseract.image_to_string(img).strip() or "Error: No readable text in image."
    except Exception as e:
        return f"Error extracting image text: {str(e)}"

def extract_text(file_path):
    """Extract text based on the file type (PDF, scanned PDF, EPUB, MOBI, or Image)."""
    try:
        file_ext = file_path.lower().split(".")[-1]

        if file_ext == 'epub':
            return extract_text_from_epub(file_path)
        elif file_ext == 'mobi':
            return extract_text_from_mobi(file_path)
        elif file_ext == 'pdf':
            text = extract_text_from_pdf(file_path)
            if not text:
                text = extract_text_from_scanned_pdf(file_path)
            return text
        elif file_ext in ['jpg', 'jpeg', 'png']:
            return extract_text_from_image(file_path)
        else:
            return f"Error: Unsupported file type ({file_ext})"
    
    except Exception as e:
        return f"Error processing file: {str(e)}"
