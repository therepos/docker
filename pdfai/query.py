import os
import json
import requests
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.chat_models import ChatOpenAI
from langchain_community.llms import Ollama
from langchain_community.embeddings import OpenAIEmbeddings

OLLAMA_URL = "http://localhost:11434"
CONFIG_FILE = "config.json"
DEFAULT_MODEL = "llama3.2"

def load_vector_store():
    """Loads stored text chunks from FAISS."""
    return FAISS.load_local("faiss_index", OpenAIEmbeddings())

def get_preferred_model():
    """Reads the preferred model from config.json or environment variable."""
    if os.getenv("OLLAMA_MODEL"):
        return os.getenv("OLLAMA_MODEL")  # Use env var if set

    try:
        with open(CONFIG_FILE, "r") as f:
            config = json.load(f)
        return config.get("ollama_model", DEFAULT_MODEL)  # Use default if not set
    except Exception:
        return DEFAULT_MODEL  # Fallback to default

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

    if use_local:
        model = get_preferred_model() or get_available_model()
        if not model:
            return "No available models found in Ollama."
        llm = Ollama(model=model)
    else:
        llm = ChatOpenAI(model="gpt-4", openai_api_key=os.getenv("OPENAI_API_KEY"))

    qa_chain = RetrievalQA.from_chain_type(llm, retriever=retriever)
    return qa_chain.run(query)
