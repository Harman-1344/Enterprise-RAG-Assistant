import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

def create_vector_db(pdf_path, persist_directory="faiss_index"):
    print(f"Loading document: {pdf_path}")
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    # Deep AI Context Handling: Split text into 1000-character chunks 
    # with a 200-character overlap so sentences aren't cut in half.
    print("Chunking text for LLM context window...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)

    # Deep AI Integration: Convert text to dense mathematical vectors using HuggingFace
    print("Generating Vector Embeddings (Downloading model on first run)...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Build and save the AI Database
    print("Building FAISS Vector Database...")
    vector_db = FAISS.from_documents(texts, embeddings)
    vector_db.save_local(persist_directory)
    
    print(f"Success! Vector Database saved to local folder: {persist_directory}")
    return True