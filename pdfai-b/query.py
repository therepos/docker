import os
import traceback 
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_ollama import OllamaLLM, OllamaEmbeddings

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "mistral")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")

def query_ai(query):
    """Retrieves relevant text from FAISS and generates a response using Ollama."""
    try:
        print(f"DEBUG: Query received: {query}")
        print(f"DEBUG: Using model '{OLLAMA_MODEL}' with base URL '{OLLAMA_BASE_URL}'")

        # Load embeddings
        embeddings = OllamaEmbeddings(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL)
        print("DEBUG: Embeddings initialized successfully.")

        # Load FAISS index
        print("DEBUG: Attempting to load FAISS index from /app/faiss_index...")
        vector_store = FAISS.load_local("/app/faiss_index", embeddings, allow_dangerous_deserialization=True)
        print("DEBUG: FAISS index loaded successfully.")

        # Retrieve relevant documents
        retriever = vector_store.as_retriever()
        print("DEBUG: FAISS retriever initialized.")

        # Initialize Ollama LLM
        llm = OllamaLLM(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL)
        print("DEBUG: Ollama LLM initialized successfully.")

        # Create query chain
        qa_chain = RetrievalQA.from_chain_type(llm, retriever=retriever)
        print("DEBUG: Query chain initialized.")

        # Execute query
        print(f"DEBUG: Executing query: {query}")
        response = qa_chain.invoke({"query": query})

        print(f"DEBUG: Response generated: {response}")
        return response

    except Exception as e:
        error_details = traceback.format_exc()
        print(f"ERROR: Query failed - {str(e)}")
        print(f"ERROR DETAILS:\n{error_details}")
        return f"Error processing query: {str(e)}"
