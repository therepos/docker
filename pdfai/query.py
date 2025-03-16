from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.chat_models import ChatOpenAI
from langchain_community.llms import Ollama
import os

def load_vector_store():
    """Loads stored text chunks from FAISS."""
    return FAISS.load_local("faiss_index", OpenAIEmbeddings())

def query_ai(query, use_local=True):
    """Retrieves relevant text and generates an answer with AI."""
    retriever = load_vector_store().as_retriever()
    
    if use_local:
        llm = Ollama(model="mistral")  # Use local model
    else:
        llm = ChatOpenAI(model="gpt-4", openai_api_key=os.getenv("OPENAI_API_KEY"))

    qa_chain = RetrievalQA.from_chain_type(llm, retriever=retriever)
    return qa_chain.run(query)
