from langchain.text_splitter import RecursiveCharacterTextSplitter

def chunk_text(text, chunk_size=1000, chunk_overlap=100):
    """Splits text into manageable chunks for AI processing."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_text(text)
