# Simple RAG Chatbot

A basic Retrieval-Augmented Generation (RAG) chatbot that answers questions based on the content of a PDF document. Built as a learning project to understand how RAG pipelines work end-to-end.

## How it works

1. **Ingestion** (`ingest.py`) — loads a PDF, splits it into chunks, generates embeddings, and stores them in a local vector database (Chroma).
2. **Retrieval + Chat** (`app.py`) — loads the existing vector database, retrieves the most relevant chunks for a user's question, and passes them to an LLM to generate a grounded answer.

```
PDF → Load → Split into chunks → Embed → Store in Chroma DB
                                                  ↓
User Question → Retriever (top-k similar chunks) → Prompt (context + question) → LLM → Answer
```

## Tech Stack

- **Framework:** [LangChain](https://www.langchain.com/)
- **LLM:** Groq (`llama-3.1-8b-instant`)
- **Embeddings:** NVIDIA NIM (`nvidia/nv-embedqa-e5-v5`)
- **Vector Store:** Chroma (local, persisted on disk)
- **PDF Loading:** PyPDFLoader

## Project Structure

```
simple_rag/
│
├── app.py              # Loads vector DB, runs retrieval + chat loop
├── ingest.py            # One-time script: PDF -> chunks -> embeddings -> chroma_db/
├── requirements.txt
├── .env.example          # Template for required API keys
│
├── data/
│   └── your_file.pdf      # Source document
│
└── chroma_db/               # Vector database (auto-generated, gitignored)
```

## Setup

1. Clone the repo and create a virtual environment:
   ```
   python -m venv venv
   venv\Scripts\activate      # Windows
   source venv/bin/activate   # Mac/Linux
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Copy `.env.example` to `.env` and add your API keys:
   ```
   GROQ_API_KEY=your_key_here
   NVIDIA_API_KEY=your_key_here
   ```
   - Get a Groq key from [console.groq.com](https://console.groq.com)
   - Get an NVIDIA key from [build.nvidia.com](https://build.nvidia.com)

4. Add your PDF to the `data/` folder.

5. Build the vector database (run once, or whenever the source PDF changes):
   ```
   python ingest.py
   ```

6. Start the chatbot:
   ```
   python app.py
   ```

## Example

```
You: what is this document about?
Bot: [answer generated from the PDF content]

You: what is the capital of France?
Bot: Mujhe iska jawab document me nahi mila.
```

The bot only answers from the provided document — it won't hallucinate answers outside the given context.

## Notes

- This project uses `langchain_classic` for chain construction (`create_retrieval_chain`, `create_stuff_documents_chain`), since these were moved out of the core `langchain` package in LangChain v1.0.
- The vector database (`chroma_db/`) is not committed to git — run `ingest.py` locally to regenerate it.
