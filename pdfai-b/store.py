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
        empty_vector_store = FAISS.from_texts(["FAISS_INIT"], embeddings, metadatas=[{"source": "init"}])
        empty_vector_store.save_local(FAISS_INDEX_PATH)
        print("DEBUG: Empty FAISS index created.")

def store_in_faiss(chunks, uid):  # <-- Change `file_id` to `uid`
    """Stores extracted text chunks in FAISS while preserving existing data, associating chunks with uid."""
    initialize_faiss()  # Ensure FAISS is initialized before trying to load

    embeddings = OllamaEmbeddings(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL)

    try:
        # Load existing FAISS index
        vector_store = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
        print("DEBUG: Existing FAISS index loaded.")

        # Associate each chunk with its uid
        metadata_entries = [{"source": uid} for _ in chunks]
        
        # Add new documents to existing index
        vector_store.add_texts(chunks, metadatas=metadata_entries)
        print(f"DEBUG: Appended {len(chunks)} new chunks from {uid} to FAISS index.")

    except Exception as e:
        print(f"ERROR: Failed to load existing FAISS index. Creating a new one. Error: {str(e)}")
        metadata_entries = [{"source": uid} for _ in chunks]
        vector_store = FAISS.from_texts(chunks, embeddings, metadatas=metadata_entries)

    # Save the updated FAISS index
    vector_store.save_local(FAISS_INDEX_PATH)
    print("DEBUG: FAISS index saved successfully.")

