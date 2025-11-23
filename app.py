import streamlit as st
import os
import tempfile
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from groq import Groq

# --- CONFIGURATION ---
# 🔴 PASTE YOUR API KEY HERE 🔴
api_key = st.secrets["GROQ_API_KEY"]
db_path = st.secrets["DB_PATH"]

client = Groq(api_key=api_key)

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "llama-3.1-8b-instant"
DB_PATH = "chroma_db_app" # Separate DB for the app to avoid conflicts

# --- PAGE SETUP ---
st.set_page_config(page_title="Legal Eagle AI", page_icon="⚖️", layout="wide")
st.title("⚖️ Legal Eagle: Chat with your Contracts")

# --- SESSION STATE ---
if "messages" not in st.session_state:
    st.session_state.messages = []

if "vector_db" not in st.session_state:
    st.session_state.vector_db = None

# --- SIDEBAR: FILE UPLOAD ---
with st.sidebar:
    st.header("📂 Document Upload")
    uploaded_file = st.file_uploader("Upload a Legal PDF", type="pdf")
    
    if uploaded_file and st.button("Process Document"):
        with st.spinner("Reading and embedding document..."):
            try:
                # 1. Save uploaded file temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_path = tmp_file.name

                # 2. Load and Split
                loader = PyPDFLoader(tmp_path)
                docs = loader.load()
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                chunks = text_splitter.split_documents(docs)
                
                # 3. Embed and Store
                embedding_function = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL)
                
                # Create a new DB in memory/disk for this session
                st.session_state.vector_db = Chroma.from_documents(
                    documents=chunks,
                    embedding=embedding_function,
                    persist_directory=DB_PATH
                )
                
                st.success(f"✅ Processed {len(chunks)} chunks!")
                os.remove(tmp_path) # Cleanup temp file

            except Exception as e:
                st.error(f"Error processing file: {e}")

# --- MAIN CHAT INTERFACE ---

# 1. Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 2. Handle User Input
if prompt := st.chat_input("Ask a question about the contract..."):
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 3. Generate Response
    if st.session_state.vector_db is None:
        with st.chat_message("assistant"):
            st.error("⚠️ Please upload and process a PDF first!")
    else:
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            try:
                # A. Retrieve Context
                results = st.session_state.vector_db.similarity_search(prompt, k=3)
                context_text = "\n\n".join([doc.page_content for doc in results])
                
                # B. Construct Prompt
                system_prompt = f"""
                You are an expert lawyer AI. Answer the user's question based ONLY on the context provided below.
                If the answer is not present, say 'I cannot find that clause in the document.'
                
                CONTEXT:
                {context_text}
                """
                
                # C. Call LLM
                client = Groq(api_key=api_key)
                stream = client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt}
                    ],
                    model=LLM_MODEL,
                    temperature=0.0,
                    stream=True # Enable streaming for cool typing effect
                )
                
                # D. Stream Output
                full_response = ""
                for chunk in stream:
                    if chunk.choices[0].delta.content:
                        full_response += chunk.choices[0].delta.content
                        message_placeholder.markdown(full_response + "▌")
                
                message_placeholder.markdown(full_response)
                
                # Add AI response to history
                st.session_state.messages.append({"role": "assistant", "content": full_response})

            except Exception as e:
                st.error(f"API Error: {e}")