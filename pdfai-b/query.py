import os
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_ollama import OllamaLLM, OllamaEmbeddings

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "mistral")

def query_ai(query):
    """Retrieves relevant text from FAISS and generates a response."""
    embeddings = OllamaEmbeddings(model=OLLAMA_MODEL)
    vector_store = FAISS.load_local("/app/faiss_index", embeddings, allow_dangerous_deserialization=True)

    retriever = vector_store.as_retriever()
    llm = OllamaLLM(model=OLLAMA_MODEL)
    qa_chain = RetrievalQA.from_chain_type(llm, retriever=retriever)

    try:
        return qa_chain.invoke({"query": query})
    except Exception as e:
        return f"Error processing query: {str(e)}"
