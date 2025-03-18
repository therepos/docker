import os
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from detect_ollama import detect_ollama_service  # Import service detection
from store import load_metadata  # Import metadata loading

# Dynamically determine Ollama host
OLLAMA_SERVICE = os.getenv("OLLAMA_SERVICE", detect_ollama_service())
OLLAMA_URL = f"http://{OLLAMA_SERVICE}:11434"

# Standard model selection
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "mistral")

def load_vector_store():
    """Loads stored text chunks from FAISS using the correct embeddings."""
    return FAISS.load_local("/app/faiss_index", OllamaEmbeddings(model=OLLAMA_MODEL), allow_dangerous_deserialization=True)

def query_ai(query, uid=None):
    """Retrieves relevant text and generates an AI answer using Ollama.
    
    - If `uid` is provided, filters results to match that UID.
    - Otherwise, searches across all uploaded documents.
    """
    vector_store = load_vector_store()
    retriever = vector_store.as_retriever()
    
    # Step 1: Fetch all relevant results
    docs = retriever.get_relevant_documents(query)

    # Step 2: If UID is provided, filter results manually
    if uid:
        metadata = load_metadata()  # Load metadata from stored JSON
        if uid not in metadata:
            return f"Error: No document found with UID {uid}"

        # Extract stored UIDs from metadata
        uids_in_metadata = set(metadata.keys())

        # Filter retrieved documents by UID
        filtered_docs = [doc for doc in docs if doc.metadata and doc.metadata.get("uid") in uids_in_metadata]

        if not filtered_docs:
            return f"No relevant results found for UID {uid}."

    else:
        filtered_docs = docs  # No filtering, return all retrieved docs

    # Step 3: Pass filtered documents to the LLM for answering
    llm = OllamaLLM(model=OLLAMA_MODEL)
    qa_chain = RetrievalQA.from_chain_type(llm, retriever=retriever)

    try:
        response = qa_chain.invoke({"query": query, "documents": filtered_docs})
        return response
    except Exception as e:
        return f"Error processing query: {str(e)}"
