import os
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_ollama import OllamaLLM  # Updated import
from langchain_ollama import OllamaEmbeddings

# Fetch Ollama host and model from environment variables
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://ollama:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "mistral")  # Default to mistral

def load_vector_store():
    """Loads stored text chunks from FAISS using the correct embeddings."""
    return FAISS.load_local("/app/faiss_index", OllamaEmbeddings(model=OLLAMA_MODEL), allow_dangerous_deserialization=True)

def query_ai(query, uid=None):
    """Retrieves relevant text and generates an AI answer using Ollama."""
    retriever = load_vector_store().as_retriever()

    if uid:
        # If UID is provided, filter by UID's embeddings
        retriever = retriever.filter_by_uid(uid)

    llm = OllamaLLM(model=OLLAMA_MODEL)
    qa_chain = RetrievalQA.from_chain_type(llm, retriever=retriever)

    try:
        # Use invoke instead of run (LangChain recommends this)
        response = qa_chain.invoke({"query": query})
        return response
    except Exception as e:
        return f"Error processing query: {str(e)}"
