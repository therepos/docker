import os
import requests
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.chat_models import ChatOpenAI
from langchain_community.llms import Ollama
from langchain_community.embeddings import OpenAIEmbeddings

OLLAMA_URL = "http://localhost:11434"

def load_vector_store():
    """Loads stored text chunks from FAISS."""
    return FAISS.load_local("faiss_index", OpenAIEmbeddings())

def get_model_from_env():
    """Fetches the model from the environment variable."""
    return os.getenv("OLLAMA_MODEL")  # Get model from environment, no default fallback

def get_available_model():
    """Fetches available models from Ollama and returns the first one."""
    try:
        response = requests.get(f"{OLLAMA_URL}/api/tags")
        response.raise_for_status()
        models = response.json().get("models", [])
        return models[0]["name"] if models else None
    except Exception:
        return None

def query_ai(query, use_local=True):
    """Retrieves relevant text and generates an AI answer."""
    retriever = load_vector_store().as_retriever()

    model = get_model_from_env()  # Get model from environment variable
    if not model:
        return "No available models found in Ollama."

    if use_local:
        llm = Ollama(model=model)
    else:
        llm = ChatOpenAI(model="gpt-4", openai_api_key=os.getenv("OPENAI_API_KEY"))

    qa_chain = RetrievalQA.from_chain_type(llm, retriever=retriever)
    return qa_chain.run(query)
