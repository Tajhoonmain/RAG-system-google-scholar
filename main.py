import os
import pandas as pd
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

load_dotenv()

# CONFIG
PDF_FOLDER = "papers"  # Ensure your 39 PDFs are in this folder
CSV_FILE = "A_02_agentic.csv"
DB_PATH = "faiss_index"

def create_vector_db():
    print("--- Step 1: Loading Metadata ---")
    df = pd.read_csv(CSV_FILE)
    
    # Create a lookup dictionary: filename -> title/url
    # Assumes PDF filenames match the ID or Title roughly, 
    # OR we just attach metadata based on the row index if order matches.
    # For this assignment, we will attach the Abstract as a "Document" too.
    
    documents = []
    
    # 1. Add Abstracts (High quality summaries)
    print("Processing CSV Abstracts...")
    for _, row in df.iterrows():
        # Create a document just from the abstract
        from langchain_core.documents import Document
        doc = Document(
            page_content=f"Title: {row['Title']}\nAbstract: {row['Abstract']}",
            metadata={"source": "metadata", "url": row['URL'], "year": row['Year']}
        )
        documents.append(doc)

    # 2. Add Full PDF Text
    print(f"Processing PDFs from {PDF_FOLDER}...")
    pdf_files = [f for f in os.listdir(PDF_FOLDER) if f.endswith('.pdf')]
    
    for pdf_file in pdf_files:
        try:
            loader = PyPDFLoader(os.path.join(PDF_FOLDER, pdf_file))
            pdf_docs = loader.load()
            documents.extend(pdf_docs)
        except Exception as e:
            print(f"Skipping {pdf_file}: {e}")

    print(f"Total raw documents: {len(documents)}")

    # 3. Chunking (Rubric requires 'Strategy')
    print("--- Step 2: Chunking ---")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", " "]
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks.")

    # 4. Embeddings & Storage
    print("--- Step 3: Embedding & Saving ---")
    # Using Gemini's optimized embedding model
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    
    vector_store = FAISS.from_documents(chunks, embeddings)
    vector_store.save_local(DB_PATH)
    print(f"Vector DB saved to {DB_PATH}")

if __name__ == "__main__":
    create_vector_db()