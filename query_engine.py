import argparse
import os
import warnings

# Suppress deprecation warnings for cleaner output
warnings.filterwarnings("ignore")

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from groq import Groq

# --- CONFIGURATION ---
DB_PATH = "chroma_db"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "llama-3.1-8b-instant"

def load_db():
    embedding_function = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return Chroma(persist_directory=DB_PATH, embedding_function=embedding_function)

def build_prompt(context_text, user_query):
    return f"""
    You are an expert lawyer bot. Use the following pieces of context to answer the user's question.
    If the answer is not in the context, say "I cannot find that information in the document."

    CONTEXT:
    {context_text}

    USER QUESTION:
    {user_query}

    ANSWER:
    """

def query_llm(prompt, api_key):
    client = Groq(api_key=api_key)
    return client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model=LLM_MODEL,
        temperature=0.0,
    )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The legal question.")
    args = parser.parse_args()
    user_query = args.query_text

    print("🔍 Searching document...")
    db = load_db()
    results = db.similarity_search(user_query, k=3)

    if not results:
        print("❌ No relevant context found in the document.")
        return

    context_text = "\n\n---\n\n".join([doc.page_content for doc in results])
    prompt = build_prompt(context_text, user_query)

    print(f"🧞‍♂️ Consulting the AI Lawyer ({LLM_MODEL})...")

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("Missing GROQ_API_KEY environment variable")

    try:
        chat_completion = query_llm(prompt, api_key)
        print("\n" + "="*50)
        print("⚖️  LEGAL EAGLE SAYS:")
        print("="*50)
        print(chat_completion.choices[0].message.content)
    except Exception as e:
        print(f"❌ API Error ({type(e).__name__}): {e}")

if __name__ == "__main__":
    main()