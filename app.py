"""Streamlit interface for the Rallis India Limited RAG chatbot."""

import streamlit as st
import tempfile
import os
import re
import time
from main_pipeline import load_vectorstore, build_chain, get_answer
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
import pdfplumber
from pathlib import Path
import base64

LOGO_PATH = "assets/rallis_logo.png"  # place your logo file here
PAGE_ICON = LOGO_PATH if Path(LOGO_PATH).exists() else "🌾"
# ---------- Page Config ----------
st.set_page_config(
    page_title="Rallis Farmer Chatbot",
    page_icon=PAGE_ICON,
    layout="wide"
)

# ---------- UI ----------
# ---------- UI ----------
st.markdown(
    f"""<div style="display:flex; align-items:center; gap:12px; margin-bottom:0.5rem;">
    <img src="data:image/png;base64,{base64.b64encode(open(LOGO_PATH,'rb').read()).decode()}" width="56">
    <h1 style="margin:0; padding:0;">Rallis Farmer Chatbot</h1>
    </div>""",
    unsafe_allow_html=True,
)
st.markdown("Ask any question about Rallis products — pesticides, dosage, crops, and more.")

def clean_text(text):
    """Normalize extracted PDF text.

    Parameters:
        text: Raw text extracted from a PDF page.

    Returns:
        Cleaned text with compact spacing and stripped outer whitespace.
    """
    text = re.sub(r' +', ' ', text)
    text = re.sub(r'\n\s*\n+', '\n', text)
    return text.strip()

def process_uploaded_pdfs(uploaded_files):
    """Extract and split text from uploaded PDF files.

    Parameters:
        uploaded_files: Streamlit-uploaded PDF files to add to the knowledge base.

    Returns:
        A list of LangChain Document objects containing chunked PDF text and metadata.
    """
    # Configure chunking for uploaded documents before embedding them.
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", " "]
    )
    all_docs = []
    for uploaded_file in uploaded_files:
        # Persist the uploaded file temporarily because pdfplumber expects a file path.
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name
        with pdfplumber.open(tmp_path) as pdf:
            for i, page in enumerate(pdf.pages):
                raw_text = page.extract_text(layout=True)
                if raw_text and raw_text.strip():
                    cleaned = clean_text(raw_text)
                    pieces = splitter.split_text(cleaned)
                    for piece in pieces:
                        all_docs.append(Document(
                            page_content=piece,
                            metadata={"page": i + 1, "source": uploaded_file.name}
                        ))
        os.unlink(tmp_path)
    return all_docs

# Sidebar
with st.sidebar:
    # Collect runtime settings and optional PDF uploads from the user.
    st.header("⚙️ Settings")
    
    temperature = st.slider("Temperature", 0.0, 1.0, 0.3, 0.1)
    
    st.markdown("---")
    
    st.subheader("📄 Upload Additional PDFs")
    uploaded_files = st.file_uploader(
        "Upload additional catalogues or documents",
        type=["pdf"],
        accept_multiple_files=True
    )

    st.markdown("---")

    st.subheader("🌐 Language")
    language = st.selectbox("Response Language", ["English", "Hindi", "Marathi"])
    
    st.markdown("---")
    
    st.subheader("🎭 Model Role")
    role_options = {
        "Farmer Assistant": """You are a simple and friendly assistant helping farmers. 
Use very simple language. Avoid technical jargon. 
Explain things as if talking to a farmer with basic education.
Use short sentences. Be warm and encouraging.""",

        "Expert Agronomist": """You are an expert agronomist with deep technical knowledge.
Provide detailed scientific information including active ingredient mechanisms, resistance management, and application best practices.
Use precise technical terminology. Include PHI (Pre Harvest Interval) and safety information.
Structure your answer with clear sections.""",

        "Sales Representative": """You are an enthusiastic Rallis India sales representative.
Highlight the key benefits and effectiveness of Rallis products.
Be confident and persuasive. Recommend specific products clearly.
Mention pack sizes and availability. End with a call to action.""",

        "Custom": ""
    }
    selected_role = st.selectbox("Select Role", list(role_options.keys()))
    
    if selected_role == "Custom":
        system_role = st.text_area("Enter custom role", placeholder="You are a...")
    else:
        system_role = role_options[selected_role]
        st.caption(f"_{role_options[selected_role][:80]}..._")

    st.markdown("---")
    if st.button("🔄 Reset Knowledge Base"):
        st.cache_resource.clear()
        st.session_state.messages = []
        st.rerun()
    st.markdown("---")
    st.markdown("**About**")
    st.markdown("Powered by Rallis India product catalogue.")

# ---------- Cache only vectorstore (slow) ----------
@st.cache_resource
def get_vectorstore():
    """Load the persisted FAISS vector store once per Streamlit session.

    Parameters:
        None.

    Returns:
        Loaded FAISS vector store used for retrieval.
    """
    return load_vectorstore()

vectorstore = get_vectorstore()

# ---------- Handle uploaded PDFs ----------
if uploaded_files:
    with st.spinner(f"Processing {len(uploaded_files)} uploaded PDF(s)..."):
        new_docs = process_uploaded_pdfs(uploaded_files)
        embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
        batch_size = 20
        extra_vectorstore = None
        # Build a temporary vector store in batches to avoid embedding rate limits.
        for i in range(0, len(new_docs), batch_size):
            batch = new_docs[i:i + batch_size]
            if extra_vectorstore is None:
                extra_vectorstore = FAISS.from_documents(batch, embeddings)
            else:
                extra_vectorstore.add_documents(batch)
            if i + batch_size < len(new_docs):
                time.sleep(10)
        vectorstore.merge_from(extra_vectorstore)
        st.sidebar.success(f"✅ {len(uploaded_files)} PDF(s) added to knowledge base ({len(new_docs)} chunks)")

# ---------- Build chain fresh on every role/language/temperature change ----------
chain, retriever = build_chain(
    vectorstore,
    temperature=temperature,
    system_role=system_role,
    language=language
)

# ---------- Chat history ----------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "current_settings" not in st.session_state:
    st.session_state.current_settings = (system_role, language)

if st.session_state.current_settings != (system_role, language):
    st.session_state.messages = []
    st.session_state.current_settings = (system_role, language)

# Display chat history
for message in st.session_state.messages:
    # Replay stored messages so the chat remains visible across reruns.
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ---------- Chat input ----------
if query := st.chat_input("Ask about a Rallis product..."):
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Searching catalogue..."):
            # Retrieve a grounded answer and the source pages used for attribution.
            answer, pages = get_answer(chain, retriever, query)

        st.markdown(answer)
        st.caption(f"📄 Source pages from catalogue: {pages}")

        st.session_state.messages.append({
            "role": "assistant",
            "content": f"{answer}\n\n📄 Source pages: {pages}"
        })
