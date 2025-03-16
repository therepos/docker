from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings

def store_in_faiss(chunks):
    """Stores extracted text chunks in FAISS vector database."""
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_texts(chunks, embeddings)
    vector_store.save_local("faiss_index")
