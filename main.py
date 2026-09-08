"""
main.py — Multi-Format RAG Chatbot
------------------------------------
In a single Streamlit app:
  - PDF (normal + unstructured/scanned)
  - CSV
  - PPT (.pptx)
  - HTML link (webpage URL)
You can upload or enter any combination of these sources to build a combined knowledge base, then chat with it.

Stack: LangChain + NVIDIA Embeddings (nemotron-3-embed-1b) + Groq LLM + Chroma (in-memory per session)
"""

import os
import tempfile
import shutil
import uuid

import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import (
    PyPDFLoader,
    UnstructuredPDFLoader,
    CSVLoader,
    UnstructuredPowerPointLoader,
    WebBaseLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain

load_dotenv()

# ----------------------------------------------------------------------
# PAGE CONFIG
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="Omni-RAG | Ask Anything",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------------------------------------------------
# "CRAZY" CUSTOM CSS — dark, neon, glassmorphism
# ----------------------------------------------------------------------
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&family=JetBrains+Mono:wght@400;600&display=swap');

html, body, [class*="css"]  {
    font-family: 'Space Grotesk', sans-serif;
}

.stApp {
    background: radial-gradient(circle at 20% 20%, #1a0b2e 0%, #0f0c1d 40%, #05040a 100%);
    color: #eae6ff;
}

/* Animated gradient title */
.omni-title {
    font-size: 2.6rem;
    font-weight: 700;
    background: linear-gradient(90deg, #ff6ec7, #7873f5, #4ade80, #ff6ec7);
    background-size: 300% 300%;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    animation: gradientMove 6s ease infinite;
    margin-bottom: 0;
}
@keyframes gradientMove {
    0% {background-position: 0% 50%;}
    50% {background-position: 100% 50%;}
    100% {background-position: 0% 50%;}
}
.omni-subtitle {
    color: #a89fd1;
    font-size: 1rem;
    margin-top: -6px;
    margin-bottom: 1.4rem;
}

/* Glass cards */
.glass-card {
    background: rgba(255, 255, 255, 0.05);
    border: 1px solid rgba(255, 255, 255, 0.12);
    border-radius: 16px;
    padding: 1rem 1.2rem;
    backdrop-filter: blur(10px);
    box-shadow: 0 8px 32px rgba(0,0,0,0.35);
    margin-bottom: 1rem;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #150c26 0%, #0a0714 100%);
    border-right: 1px solid rgba(255,255,255,0.08);
}
section[data-testid="stSidebar"] .stMarkdown h2,
section[data-testid="stSidebar"] .stMarkdown h3 {
    color: #ff9fe5;
}

/* Buttons */
.stButton>button, .stDownloadButton>button {
    background: linear-gradient(90deg, #7873f5, #ff6ec7);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 0.6rem 1.2rem;
    font-weight: 600;
    transition: transform 0.15s ease, box-shadow 0.15s ease;
}
.stButton>button:hover, .stDownloadButton>button:hover {
    transform: translateY(-2px) scale(1.02);
    box-shadow: 0 6px 20px rgba(120,115,245,0.45);
}

/* Chat bubbles */
.chat-user {
    background: linear-gradient(135deg, #4338ca, #7873f5);
    color: white;
    padding: 0.8rem 1.1rem;
    border-radius: 18px 18px 4px 18px;
    margin: 0.4rem 0;
    max-width: 80%;
    margin-left: auto;
    box-shadow: 0 4px 14px rgba(67,56,202,0.35);
}
.chat-bot {
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.1);
    color: #f1edff;
    padding: 0.8rem 1.1rem;
    border-radius: 18px 18px 18px 4px;
    margin: 0.4rem 0;
    max-width: 80%;
    box-shadow: 0 4px 14px rgba(0,0,0,0.3);
}

/* Badges for loaded sources */
.src-badge {
    display: inline-block;
    background: rgba(74, 222, 128, 0.15);
    color: #4ade80;
    border: 1px solid rgba(74, 222, 128, 0.4);
    border-radius: 999px;
    padding: 0.2rem 0.7rem;
    font-size: 0.78rem;
    margin: 0.15rem;
    font-family: 'JetBrains Mono', monospace;
}

hr {border-color: rgba(255,255,255,0.08);}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ----------------------------------------------------------------------
# SESSION STATE
# ----------------------------------------------------------------------
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())[:8]
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "loaded_sources" not in st.session_state:
    st.session_state.loaded_sources = []


# ----------------------------------------------------------------------
# LOADERS — one function per format; all return List[Document]
# ----------------------------------------------------------------------
def load_pdf(path: str, unstructured: bool):
    if unstructured:
        loader = UnstructuredPDFLoader(path)
    else:
        loader = PyPDFLoader(path)
    return loader.load()


def load_csv(path: str):
    loader = CSVLoader(path)
    return loader.load()


def load_ppt(path: str):
    loader = UnstructuredPowerPointLoader(path)
    return loader.load()


def load_html(url: str):
    loader = WebBaseLoader(url)
    return loader.load()


def save_upload_to_tmp(uploaded_file, tmp_dir: str) -> str:
    path = os.path.join(tmp_dir, uploaded_file.name)
    with open(path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return path


# ----------------------------------------------------------------------
# PIPELINE — build the vector store and RAG chain from the provided sources
# ----------------------------------------------------------------------
def build_knowledge_base(uploaded_files, html_urls, unstructured_pdf, progress_cb=None):
    all_docs = []
    sources_loaded = []
    tmp_dir = tempfile.mkdtemp(prefix="omni_rag_")

    try:
        # --- uploaded files (PDF / CSV / PPTX) ---
        for f in uploaded_files or []:
            ext = f.name.lower().split(".")[-1]
            if progress_cb:
                progress_cb(f"Reading {f.name} ...")
            saved_path = save_upload_to_tmp(f, tmp_dir)
            try:
                if ext == "pdf":
                    docs = load_pdf(saved_path, unstructured_pdf)
                elif ext == "csv":
                    docs = load_csv(saved_path)
                elif ext in ("ppt", "pptx"):
                    docs = load_ppt(saved_path)
                else:
                    st.warning(f"Skipped unsupported file: {f.name}")
                    continue
                all_docs.extend(docs)
                sources_loaded.append(f.name)
            except Exception as e:
                st.error(f"Failed to load {f.name}: {e}")

        # --- HTML links, one or multiple, newline separated ---
        for url in html_urls or []:
            url = url.strip()
            if not url:
                continue
            if progress_cb:
                progress_cb(f"Fetching {url} ...")
            try:
                docs = load_html(url)
                all_docs.extend(docs)
                sources_loaded.append(url)
            except Exception as e:
                st.error(f"Failed to load {url}: {e}")

        if not all_docs:
            return None, None, sources_loaded

        if progress_cb:
            progress_cb("Splitting into chunks ...")
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        chunks = splitter.split_documents(all_docs)

        if progress_cb:
            progress_cb(f"Embedding {len(chunks)} chunks ...")
        embeddings = NVIDIAEmbeddings(
            model="nvidia/nemotron-3-embed-1b",
            api_key=os.getenv("NVIDIA_API_KEY"),
        )

        # Fresh in-memory collection per session so reruns do not mix data
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            collection_name=f"omni_{st.session_state.session_id}",
        )

        retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

        llm = ChatGroq(
            model="openai/gpt-oss-120b",
            temperature=0.2,
            api_key=os.getenv("GROQ_API"),
        )

        prompt = ChatPromptTemplate.from_template(
            """You are a helpful assistant. Answer only using the context given below.
            If the answer isn't in the context, clearly say
            "I could not find any answer in the document."

            context : {context}

            question : {input}

            answer :"""
        )

        document_chain = create_stuff_documents_chain(llm, prompt)
        rag_chain = create_retrieval_chain(retriever, document_chain)

        if progress_cb:
            progress_cb("Done!")

        return vectorstore, rag_chain, sources_loaded

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# ----------------------------------------------------------------------
# SIDEBAR — ingestion controls
# ----------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 📥 Load Your Sources")
    st.caption("Upload one or multiple files — use only the sources you need.")

    uploaded_files = st.file_uploader(
        "PDF / CSV / PPTX (multiple allowed)",
        type=["pdf", "csv", "ppt", "pptx"],
        accept_multiple_files=True,
    )

    unstructured_pdf = st.checkbox(
        "Treat PDFs as scanned / unstructured (OCR-style parsing)",
        value=False,
        help="Enable this for image-heavy or messy-layout PDFs.",
    )

    st.markdown("---")
    html_input = st.text_area(
        "HTML link(s) — one URL per line",
        placeholder="https://example.com/article\nhttps://another-site.com/page",
        height=90,
    )
    html_urls = html_input.splitlines() if html_input else []

    st.markdown("---")
    build_clicked = st.button("🚀 Build Knowledge Base", use_container_width=True)

    if build_clicked:
        if not uploaded_files and not any(u.strip() for u in html_urls):
            st.warning("Please upload at least one file or provide a link.")
        else:
            status_box = st.empty()

            def progress_cb(msg):
                status_box.info(msg)

            with st.spinner("Building knowledge base..."):
                vectorstore, rag_chain, sources = build_knowledge_base(
                    uploaded_files, html_urls, unstructured_pdf, progress_cb
                )

            if rag_chain is None:
                st.error("Nothing could be loaded. Please check your files and links.")
            else:
                st.session_state.vectorstore = vectorstore
                st.session_state.rag_chain = rag_chain
                st.session_state.loaded_sources = sources
                st.session_state.chat_history = []
                status_box.success("Knowledge base ready! You can start chatting 👉")

    if st.session_state.loaded_sources:
        st.markdown("### ✅ Loaded")
        badges = "".join(
            f'<span class="src-badge">{s}</span>' for s in st.session_state.loaded_sources
        )
        st.markdown(badges, unsafe_allow_html=True)

    if st.session_state.rag_chain and st.button("🗑️ Reset Session", use_container_width=True):
        st.session_state.vectorstore = None
        st.session_state.rag_chain = None
        st.session_state.chat_history = []
        st.session_state.loaded_sources = []
        st.rerun()


# ----------------------------------------------------------------------
# MAIN AREA — chat
# ----------------------------------------------------------------------
st.markdown('<div class="omni-title">🧠 Omni-RAG</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="omni-subtitle">PDF · Scanned PDF · CSV · PPT · Webpages — ask questions about everything in one place.</div>',
    unsafe_allow_html=True,
)

if not st.session_state.rag_chain:
    st.markdown(
        '<div class="glass-card">👈 Upload a file or add a link from the sidebar, then '
        '<b>Build Knowledge Base</b> to start chatting here.</div>',
        unsafe_allow_html=True,
    )
else:
    for turn in st.session_state.chat_history:
        role_class = "chat-user" if turn["role"] == "user" else "chat-bot"
        st.markdown(f'<div class="{role_class}">{turn["content"]}</div>', unsafe_allow_html=True)

    user_query = st.chat_input("Ask a question about your sources...")

    if user_query:
        st.session_state.chat_history.append({"role": "user", "content": user_query})
        st.markdown(f'<div class="chat-user">{user_query}</div>', unsafe_allow_html=True)

        with st.spinner("Thinking... 🤔"):
            try:
                response = st.session_state.rag_chain.invoke({"input": user_query})
                answer = response["answer"]
            except Exception as e:
                answer = f"Error: {e}"

        st.session_state.chat_history.append({"role": "assistant", "content": answer})
        st.markdown(f'<div class="chat-bot">{answer}</div>', unsafe_allow_html=True)