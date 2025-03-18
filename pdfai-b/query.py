import os
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from store import load_metadata  # Import metadata

# Load Ollama service name from environment variables (set in docker-compose)
OLLAMA_SERVICE = os.getenv("OLLAMA_SERVICE", "ollama")
OLLAMA_URL = f"http://{OLLAMA_SERVICE}:11434"

# Standard model selection
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "mistral")

def query_ai(query, uid=None):
    """Retrieves relevant text and generates an AI answer using Ollama.
    
    - If `uid` is provided, retrieves only the specific document's embeddings using FAISS filtering.
    - Otherwise, searches across all uploaded documents.
    """
    # Initialize Ollama embeddings
    embeddings = OllamaEmbeddings(model=OLLAMA_MODEL)
    
    # Load the vector store (FAISS)
    vector_store = FAISS.load_local("/app/faiss_index", embeddings, allow_dangerous_deserialization=True)

    # Debug print
    print(f"DEBUG: Connecting to Ollama at {OLLAMA_URL} with model {OLLAMA_MODEL}")

    metadata = load_metadata()

    # If UID is provided, filter FAISS results by UID
    if uid:
        if uid not in metadata:
            return f"Error: No document found with UID {uid}"

        # Use FAISS filtering to retrieve only relevant docs for the UID
        retriever = vector_store.as_retriever(search_kwargs={"filter": {"uid": str(uid)}})

    else:
        # Retrieve results from all documents if UID is not provided
        retriever = vector_store.as_retriever()

    # Use Ollama for answering queries
    llm = OllamaLLM(model=OLLAMA_MODEL)
    qa_chain = RetrievalQA.from_chain_type(llm, retriever=retriever)

    try:
        response = qa_chain.invoke({"query": query})
        return response
    except Exception as e:
        # Debug
        print(f"DEBUG: Ollama connection failed - {str(e)}")
        return f"Error processing query: {str(e)}"
