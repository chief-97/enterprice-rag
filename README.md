# Enterprise RAG Pipeline

A production-grade Retrieval-Augmented Generation (RAG) system that ingests multiple document formats (PDF, CSV, PPTX) and web URLs, then answers questions with context-aware, source-grounded responses using a hybrid retrieval pipeline.

🔗 **Live Demo:** _[add your Streamlit Cloud link here after deployment]_

---

## ✨ Features

- 📄 Multi-format document ingestion — PDF, CSV, PPT/PPTX, and web URLs
- 🔍 Hybrid retrieval — combines dense vector search with BM25 keyword search via `EnsembleRetriever` for more accurate results
- ☁️ Cloud-native vector storage — powered by Qdrant Cloud (no local DB dependency)
- ⚡ Fast LLM inference via Groq
- 💬 Persistent chat history and session-based state management
- 🖥️ Clean, interactive Streamlit UI with real-time ingestion progress
- 🔐 Secure secrets management via environment variables

---

## 🛠️ Tech Stack

| Component        | Technology                              |
|-------------------|------------------------------------------|
| Framework         | LangChain                                |
| Vector Store      | Qdrant Cloud                             |
| Embeddings        | NVIDIA NIM — `nvidia/nv-embedqa-e5-v5`   |
| LLM               | Groq — `openai/gpt-oss-120b`             |
| Retrieval         | Hybrid (Dense + BM25 via EnsembleRetriever) |
| Frontend          | Streamlit                                |
| Language          | Python                                   |

---

## 🏗️ Architecture

```
Document Upload / URL Input
        │
        ▼
  Format-specific Loader (PDF / CSV / PPTX / HTML)
        │
        ▼
   Chunking & Preprocessing
        │
        ▼
   NVIDIA Embeddings (nv-embedqa-e5-v5)
        │
        ▼
   Qdrant Cloud Vector Store
        │
        ▼
  Hybrid Retriever (Dense + BM25)
        │
        ▼
   Groq LLM (openai/gpt-oss-120b)
        │
        ▼
   Answer + Source Citations → Streamlit UI
```

---

## 📸 Screenshots

_[Add 1-2 screenshots of the chat interface here]_

---

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- A Qdrant Cloud account and API key
- A Groq API key
- An NVIDIA NIM API key

### Installation

```bash
git clone https://github.com/chief-97/enterprice-rag.git
cd enterprice-rag
python -m venv venv
venv\Scripts\activate      # Windows
# source venv/bin/activate  # macOS/Linux
pip install -r requirements.txt
```

### Environment Variables

Copy `env.example` to `.env` and fill in your keys:

```
QDRANT_URL=your_qdrant_cloud_url
QDRANT_API_KEY=your_qdrant_api_key
GROQ_API_KEY=your_groq_api_key
NVIDIA_API_KEY=your_nvidia_nim_api_key
```

### Run the app

```bash
streamlit run main.py
```

---

## 📁 Project Structure

```
enterprice-rag/
├── main.py           # Streamlit app entry point & UI
├── app.py            # Application logic
├── ingest.py         # Document ingestion & processing pipeline
├── requirements.txt  # Python dependencies
├── env.example        # Example environment variables
└── README.md
```

---

## 📌 Roadmap / Future Improvements

- [ ] Add support for more document formats (DOCX, XLSX)
- [ ] Add re-ranking for improved retrieval accuracy
- [ ] Add authentication for multi-user support
- [ ] Add evaluation metrics (RAGAS) for answer quality

---

## 📄 License

This project is licensed under the MIT License.

---

## 🙋 Author

**Shivam Vivekanand Upadhyay**
[GitHub](https://github.com/chief-97)