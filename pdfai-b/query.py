import os
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_ollama import OllamaLLM, OllamaEmbeddings

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "mistral")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")

def query_ai(query):
    """Retrieves relevant text from FAISS and generates a response using Ollama."""
    embeddings = OllamaEmbeddings(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL)
    vector_store = FAISS.load_local("/app/faiss_index", embeddings, allow_dangerous_deserialization=True)

    retriever = vector_store.as_retriever()
    llm = OllamaLLM(model=OLLAMA_MODEL)
    qa_chain = RetrievalQA.from_chain_type(llm, retriever=retriever)

    try:
        return qa_chain.invoke({"query": query})
    except Exception as e:
        return f"Error processing query: {str(e)}"
