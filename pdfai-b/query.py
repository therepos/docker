import os
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from detect_ollama import detect_ollama_service  # Import service detection

# Dynamically determine Ollama host
OLLAMA_SERVICE = os.getenv("OLLAMA_SERVICE", detect_ollama_service())
OLLAMA_URL = f"http://{OLLAMA_SERVICE}:11434"

# Standard model selection
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "mistral")

def load_vector_store():
    """Loads stored text chunks from FAISS using the correct embeddings."""
    return FAISS.load_local("/app/faiss_index", OllamaEmbeddings(model=OLLAMA_MODEL), allow_dangerous_deserialization=True)

def query_ai(query, uid=None):
    """Retrieves relevant text and generates an AI answer using Ollama."""
    vector_store = load_vector_store()
    retriever = vector_store.as_retriever()

    if uid:
        retriever = retriever.filter_by_uid(uid)

    llm = OllamaLLM(model=OLLAMA_MODEL)
    qa_chain = RetrievalQA.from_chain_type(llm, retriever=retriever)

    try:
        response = qa_chain.invoke({"query": query})
        return response
    except Exception as e:
        return f"Error processing query: {str(e)}"
