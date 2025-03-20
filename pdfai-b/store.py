import os
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS

# Fetch model and base URL from environment variables
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "mistral")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")
FAISS_INDEX_PATH = "/app/faiss_index"

def initialize_faiss():
    """Creates an empty FAISS index if none exists."""
    embeddings = OllamaEmbeddings(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL)

    if not os.path.exists(FAISS_INDEX_PATH) or not os.path.exists(f"{FAISS_INDEX_PATH}/index.faiss"):
        print("DEBUG: FAISS index missing. Creating an empty FAISS index.")
        os.makedirs(FAISS_INDEX_PATH, exist_ok=True)
        empty_vector_store = FAISS.from_texts(["FAISS_INIT"], embeddings)
        empty_vector_store.save_local(FAISS_INDEX_PATH)
        print("DEBUG: Empty FAISS index created.")

def store_in_faiss(chunks):
    """Stores extracted text chunks in FAISS vector database while preserving existing data."""
    initialize_faiss()  # Ensure FAISS is initialized before trying to load

    embeddings = OllamaEmbeddings(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL)

    try:
        # Load existing FAISS index
        vector_store = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
        print("DEBUG: Existing FAISS index loaded.")
        
        # Add new documents to existing index
        vector_store.add_texts(chunks)
        print(f"DEBUG: Appended {len(chunks)} new chunks to FAISS index.")

    except Exception as e:
        print(f"ERROR: Failed to load existing FAISS index. Creating a new one. Error: {str(e)}")
        vector_store = FAISS.from_texts(chunks, embeddings)

    # Save the updated FAISS index
    vector_store.save_local(FAISS_INDEX_PATH)
    print("DEBUG: FAISS index saved successfully.")
