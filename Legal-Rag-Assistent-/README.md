⚖️ LegalRAG — Indian Evidence Act Research Assistant

Production-Grade Retrieval-Augmented Generation (RAG) System for Indian Law

AI-powered legal research system that enables accurate, citation-backed answers from Indian legal documents such as the Indian Evidence Act, IPC, CrPC, CPC, and related statutes — without hallucination outside uploaded documents.

🎯 Problem Statement

Legal research is:

⏳ Time-consuming

❌ Error-prone

📚 Fragmented across multiple acts & sections

Manual section lookup (e.g., “Section 58 Evidence Act”) often leads to missed context or incorrect interpretation.

💡 Solution

LegalRAG uses Retrieval-Augmented Generation (RAG) to:

Search across thousands of legal sections

Retrieve only relevant chunks

Generate strictly context-based answers

Provide verifiable sources for every response

🛑 Zero hallucination policy If the answer is not present in uploaded documents →

“Not available in the uploaded documents.”

🚀 Core Features

✅ Section-wise legal question answering ✅ Supports Indian Acts (Evidence Act, IPC, CrPC, CPC) ✅ HuggingFace / Groq / Hybrid LLM providers ✅ ChromaDB persistent vector storage ✅ Strict context-only answering ✅ CLI + Streamlit UI ready ✅ Production-ready modular architecture ✅ Chat history isolation (new chat ≠ old history)

🧠 RAG Pipeline (High Level) User Query ↓ Semantic Retriever (ChromaDB) ↓ Relevant Legal Chunks ↓ LLM (Groq / HF / Hybrid) ↓ Answer + Sources

🏗️ Project Structure (Production-Grade)
legalrag/
│
├── config/
│   ├── __init__.py
│   └── settings.py
│
├── data/
│   ├── uploads/
│   └── chroma_db/
│
├── src/
│   ├── __init__.py
│   │
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── document_processor.py
│   │   ├── run_ingestion.py
│   │   └── vector_store.py
│   │
│   ├── retrieval/
│   │   ├── __init__.py
│   │   └── retriever.py
│   │
│   ├── generation/
│   │   ├── __init__.py
│   │   └── rag_pipeline.py
│   │
│   ├── providers/
│   │   ├── __init__.py
│   │   ├── factory.py
│   │   ├── groq_provider.py
│   │   ├── huggingface_provider.py
│   │   └── hybrid_provider.py
│   │
│   ├── evaluation/
│   │   ├── __init__.py
│   │   └── metrics.py
│   │
│   ├── ui/
│   │   ├── __init__.py
│   │   └── streamlit_app.py
│   │
│   └── utils/
│       └── __init__.py
│
├── app.py
├── check_chroma.py
├── data_cleaning.py
├── requirements.txt
├── .env
└── README.md

✅ Every folder contains init.py for stable imports & production readiness

🛠️ Technology Stack Component Technology Language Python 3.10 RAG Framework LangChain Vector DB ChromaDB (Persistent) Embeddings HuggingFace Sentence Transformers LLMs Groq / HuggingFace / Hybrid UI Streamlit Config Pydantic Settings Logging Loguru ⚙️ Installation 1️⃣ Create Environment conda create -n legalrag310 python=3.10 conda activate legalrag310

2️⃣ Install Dependencies pip install -r requirements.txt

3️⃣ Configure Environment

Create .env file:

API_PROVIDER=groq GROQ_API_KEY=your_key_here

CHROMA_PERSIST_DIRECTORY=./data/chroma_db CHROMA_COLLECTION_NAME=legal_documents

📥 Ingest Legal Documents

Place .txt / .pdf files inside:

data/uploads/

Run ingestion:

python src/ingestion/run_ingestion.py

Verify storage:

python check_chroma.py

🔍 Ask Legal Questions (CLI) python -m src.generation.rag_pipeline

Example:

Ask: Explain Section 58 of the Indian Evidence Act

Output:

ANSWER: Facts admitted need not be proved. If parties admit a fact in writing or in court, no further proof is required.

SOURCES: Evidence_Act_1872_p29_c0.txt | chunk 3

🖥️ Run Streamlit UI streamlit run app.py

Features:

New chat = fresh history

Section-wise search

Source traceability

📊 Evaluation & Reliability

Context relevance enforced

Duplicate chunk filtering

Strict refusal when content missing

No external knowledge injection

🚀 Deployment Ready

✅ Streamlit Cloud

✅ HuggingFace Spaces

✅ Local production

✅ Modular provider switching

🧑‍💼 Interview Value

This project demonstrates:

Real RAG architecture

Production-ready Python

Legal domain understanding

Vector DB design

Prompt safety & hallucination control

📄 License

MIT License

🙌 Acknowledgements

LangChain

ChromaDB

HuggingFace

Groq

Indian Legal Open Data

🧠 Future Improvements

Section-aware retriever (Section 58 → exact match)

Multi-Act filtering

Citation highlighting

Answer confidence scoring

PDF upload via UI
