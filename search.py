import argparse
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# --- CONFIGURATION ---
DB_PATH = "chroma_db"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

def main():
    # 1. Parse Command Line Arguments
    # This lets us run: python search.py "My question here"
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The question you want to ask the document.")
    args = parser.parse_args()
    query_text = args.query_text

    # 2. Load the Database
    # We must use the SAME embedding function as we did during ingestion.
    print("🧠 Loading embedding model...")
    embedding_function = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL)
    
    print(f"📂 Opening database at {DB_PATH}...")
    db = Chroma(persist_directory=DB_PATH, embedding_function=embedding_function)

    # 3. Perform Semantic Search
    # k=3 means "Give me the top 3 most relevant chunks"
    print(f"🔍 Searching for: '{query_text}'")
    results = db.similarity_search_with_score(query_text, k=3)

    # 4. Display Results
    print("\n--- 📄 FOUND DOCUMENTS ---")
    for i, (doc, score) in enumerate(results):
        # Score is usually distance (lower is better for Chroma/Euclidean)
        print(f"\n🔹 Result {i+1} (Score: {score:.4f})")
        print(f"   Path: {doc.metadata.get('source', 'Unknown')}")
        print(f"   Content: {doc.page_content[:300]}...") # Show first 300 chars
        print("-" * 40)

if __name__ == "__main__":
    main()