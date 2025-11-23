import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma

# --- CONFIGURATION ---
DATA_PATH = "data"
DB_PATH = "chroma_db"
# We use a standard open-source embedding model (small and fast)
EMBEDDING_MODEL = "all-MiniLM-L6-v2" 

def ingest_documents():
    # 1. Check for PDFs
    if not os.path.exists(DATA_PATH):
        os.makedirs(DATA_PATH)
        print(f"📁 Created {DATA_PATH} folder. Please put your PDFs there!")
        return

    pdf_files = [f for f in os.listdir(DATA_PATH) if f.endswith('.pdf')]
    if not pdf_files:
        print("⚠️ No PDFs found in 'data' folder.")
        return

    all_chunks = []

    # 2. Load and Split PDFs
    print(f"📥 Found {len(pdf_files)} PDF(s). Processing...")
    
    for pdf_file in pdf_files:
        file_path = os.path.join(DATA_PATH, pdf_file)
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        
        # Splitter: Breaks text into 1000-character chunks with 200 char overlap
        # This ensures context isn't lost at the cut points.
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        chunks = text_splitter.split_documents(documents)
        all_chunks.extend(chunks)
        print(f"   📄 {pdf_file}: Split into {len(chunks)} text chunks.")

    # 3. Initialize Embedding Model
    # This will download the model weights the first time it runs (~80MB)
    print(f"🧠 Loading embedding model ({EMBEDDING_MODEL})...")
    embedding_function = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL)

    # 4. Create/Update Vector Database
    print("💾 Saving to ChromaDB...")
    
    # This creates the DB on disk and automatically embeds the chunks
    db = Chroma.from_documents(
        documents=all_chunks,
        embedding=embedding_function,
        persist_directory=DB_PATH
    )
    
    # (Optional) Persist call is often automatic in newer versions, but good practice
    db.persist()
    
    print(f"🎉 Success! Saved {len(all_chunks)} chunks to {DB_PATH}.")

if __name__ == "__main__":
    ingest_documents()