# 🧠 Self-Healing RAG Pipeline

> A production-ready Retrieval-Augmented Generation system that **critiques its own outputs and self-corrects** using LangGraph's cyclical workflows — so it never hallucinates silently.

![Python](https://img.shields.io/badge/Python-3.11-blue) ![FastAPI](https://img.shields.io/badge/FastAPI-0.115-green) ![LangGraph](https://img.shields.io/badge/LangGraph-0.2-purple) ![ChromaDB](https://img.shields.io/badge/ChromaDB-0.5-orange) ![Docker](https://img.shields.io/badge/Docker-ready-blue)

---

## 🤔 What Problem Does This Solve?

Standard RAG pipelines retrieve documents, generate an answer, and return it — no questions asked. If the retrieved chunks don't actually support the answer, the LLM hallucinates and you'd never know.

**Self-Healing RAG adds a Critic Agent** that evaluates every answer before returning it. If the answer isn't grounded in the retrieved context, the pipeline automatically reformulates the query and retries — up to 3 times — before gracefully admitting it doesn't have enough information.

---

## 🔄 How It Works

```
User Query
    │
    ▼
┌─────────────┐
│   Retrieve  │  ← ChromaDB vector search (sentence-transformers embeddings)
└──────┬──────┘
       │ top-k chunks
       ▼
┌─────────────┐
│  Generate   │  ← LLM answers using ONLY retrieved context
└──────┬──────┘
       │ answer
       ▼
┌─────────────┐
│Critic Agent │  ← LLM checks: "Is this answer grounded in the context?"
└──────┬──────┘
       │
   Grounded?
   ┌───┴───┐
  YES      NO (retry < 3)
   │        │
   │        └──► Reformulate query ──► back to Retrieve
   ▼
Final Answer                    (if retry >= 3: graceful fallback)
```

---

## ✨ Features

- 🔄 **Self-healing retrieval loop** — retries up to 3 times with reformulated queries
- 🧠 **LLM-powered critic agent** — detects hallucinations before they reach the user
- 📚 **ChromaDB vector store** — persistent local embeddings with sentence-transformers
- ⚡ **Real-time feedback** — frontend shows retries, groundedness, and source chunks
- 🚀 **FastAPI backend** — auto-generated Swagger docs at `/docs`
- 🐳 **Fully Dockerized** — one command to run anywhere
- 💸 **Free to run** — uses Groq's free tier (llama-3.1-8b-instant)
- 📄 **Document ingestion API** — add your own docs via the UI or API

---

## 🛠 Tech Stack

| Layer | Technology |
|-------|-----------|
| Orchestration | LangGraph (stateful cyclical graph) |
| LLM | Groq — llama-3.1-8b-instant (free tier) |
| Vector DB | ChromaDB (local persistent) |
| Embeddings | sentence-transformers — all-MiniLM-L6-v2 (runs locally) |
| Backend | FastAPI + Uvicorn |
| Frontend | HTML / CSS / Vanilla JS |
| Deployment | Docker + docker-compose |

---

## 🚀 Quick Start (Docker)

```bash
# 1. Clone the repo
git clone https://github.com/yourusername/self-healing-rag
cd self-healing-rag

# 2. Set up environment
cp .env.example .env
# Edit .env and add your Groq API key (free at https://console.groq.com)

# 3. Run everything
docker-compose up --build

# 4. Open the app
open http://localhost:8000/app
```

---

## 💻 Local Development (without Docker)

```bash
# 1. Clone and enter project
git clone https://github.com/yourusername/self-healing-rag
cd self-healing-rag

# 2. Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up environment
cp .env.example .env
# Add your Groq API key to .env

# 5. Ingest sample documents
python -m backend.ingest

# 6. Start the API server
python -m backend.main

# 7. Open the app
open http://localhost:8000/app
```

---

## 🔌 API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | API info |
| `GET` | `/health` | Health check + document count |
| `POST` | `/query` | Main RAG query with self-healing |
| `POST` | `/ingest` | Add new documents to vector DB |
| `GET` | `/app` | Serve frontend UI |
| `GET` | `/docs` | Auto-generated Swagger UI |

### Example Query

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "How does the TLS handshake work?"}'
```

```json
{
  "final_answer": "The TLS handshake works through a multi-step negotiation...",
  "critique": "The answer directly matches the explanation in the context.",
  "retry_count": 1,
  "retrieved_chunks": ["HTTPS/TLS handshake establishes..."],
  "is_grounded": true,
  "query": "How does the TLS handshake work?"
}
```

---

## 📁 Project Structure

```
self-healing-rag/
├── backend/
│   ├── main.py          # FastAPI app + all endpoints
│   ├── rag_pipeline.py  # LangGraph pipeline (the brain)
│   ├── retriever.py     # ChromaDB vector store logic
│   ├── critic_agent.py  # Hallucination checker
│   ├── config.py        # Environment config
│   └── ingest.py        # Document ingestion script
├── frontend/
│   └── index.html       # Single-file frontend (easy to restyle)
├── data/
│   └── sample_docs.txt  # Sample documents (5 topics)
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── .env.example
└── README.md
```

---

## 🧪 What is a Critic Agent?

A Critic Agent is a second LLM call that acts as a fact-checker for the first one.

After the generator LLM produces an answer, the critic receives the original question, the retrieved context chunks, and the generated answer. It then responds with a structured JSON verdict: was this answer actually supported by the context, or did the LLM invent details?

If the answer fails the check, the critic also produces a **reformulated search query** — a smarter way to ask the vector database for better chunks. This creates the self-healing loop: retrieve → generate → critique → re-retrieve → repeat.

This pattern is inspired by research on self-RAG and corrective RAG, implemented here using LangGraph's conditional edges to model the cyclical workflow cleanly.

---

## 🔑 Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `GROQ_API_KEY` | Groq API key (free at console.groq.com) | required |
| `OPENAI_API_KEY` | OpenAI key (optional fallback) | optional |
| `MODEL_NAME` | LLM model to use | llama-3.1-8b-instant |
| `EMBEDDING_MODEL` | Sentence-transformers model | all-MiniLM-L6-v2 |
| `CHROMA_PERSIST_DIR` | Where to store vector DB | ./chroma_db |
| `MAX_RETRIES` | Max self-healing attempts | 3 |
| `USE_GROQ` | Use Groq instead of OpenAI | true |

---

## 📝 Adding Your Own Documents

Via the UI: click **"Add Documents"** at the bottom of the page, paste your text, and hit Ingest.

Via the API:
```bash
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{"texts": ["Your document text here..."]}'
```

Via the ingest script (for bulk loading):
```bash
# Edit data/sample_docs.txt with your content, then:
python -m backend.ingest
```

---

## 🤝 Contributing

PRs welcome! Some ideas for extensions:
- Add PDF ingestion support
- Stream the answer token by token
- Add a reranker between retrieval and generation
- Support multiple collections / namespaces
- Add evaluation metrics (RAGAS)

---

## 📄 License

MIT License — use this however you like.

---

*Built with LangGraph, ChromaDB, FastAPI, and Groq.*