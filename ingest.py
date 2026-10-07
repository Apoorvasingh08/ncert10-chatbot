import os
from langchain_community.document_loaders import DirectoryLoader, PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

def ingest_data():
    print("Loading PDFs with PyMuPDF...")
    loader = DirectoryLoader("ncert_books", glob="*.pdf", loader_cls=PyMuPDFLoader)
    docs = loader.load()
    
    print(f"Loaded {len(docs)} pages. Splitting text...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    
    print("Generating free local embeddings and building FAISS index...")
    # Switched to free local embeddings
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(splits, embeddings)
    vectorstore.save_local("faiss_index")
    print("Done! Vector store saved successfully.")

if __name__ == "__main__":
    ingest_data()