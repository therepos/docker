import os
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS

# Fetch model and base URL from environment variables
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "mistral")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")
FAISS_INDEX_PATH = "/app/faiss_index"

def store_in_faiss(chunks):
    """Stores extracted text chunks in FAISS vector database while preserving existing data."""
    embeddings = OllamaEmbeddings(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL)

    # Check if FAISS index already exists
    if os.path.exists(FAISS_INDEX_PATH):
        try:
            # Load existing FAISS index safely
            vector_store = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
            print("DEBUG: Existing FAISS index loaded.")
            
            # Add new documents to existing index
            vector_store.add_texts(chunks)
            print(f"DEBUG: Appended {len(chunks)} new chunks to FAISS index.")

        except Exception as e:
            print(f"ERROR: Failed to load existing FAISS index. Creating a new one. Error: {str(e)}")
            vector_store = FAISS.from_texts(chunks, embeddings)
    else:
        # Create a new FAISS index if none exists
        print("DEBUG: No existing FAISS index found. Creating a new one.")
        vector_store = FAISS.from_texts(chunks, embeddings)

    # Save the updated FAISS index
    vector_store.save_local(FAISS_INDEX_PATH)
    print("DEBUG: FAISS index saved successfully.")
