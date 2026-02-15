⚖️ LegalRAG — Indian Evidence Act Research Assistant
Production-Grade Retrieval-Augmented Generation (RAG) System for Indian Law

AI-powered legal research system that provides accurate, citation-backed answers strictly from uploaded Indian legal documents such as:

Indian Evidence Act

IPC

CrPC

CPC

Other Indian statutes

🚫 No hallucinations.
📄 No external knowledge.
📌 Only document-grounded answers.

🎯 Problem Statement

Legal research in India is:

⏳ Time-consuming

❌ Prone to misinterpretation

📚 Fragmented across multiple Acts & Sections

🔍 Dependent on manual section lookup

Example:
Searching for “Section 58 Evidence Act” manually may miss related context or judicial interpretation.

💡 Solution — LegalRAG

LegalRAG uses Retrieval-Augmented Generation (RAG) to:

Search across thousands of legal sections

Retrieve only the most relevant chunks

Generate strictly context-based answers

Provide verifiable document citations

Enforce zero hallucination policy

If answer is not found in uploaded documents:

“Not available in the uploaded documents.”

🌟 Core Features

✅ Section-wise legal question answering
✅ Supports Indian Acts (Evidence Act, IPC, CrPC, CPC)
✅ HuggingFace / Groq / Hybrid LLM providers
✅ ChromaDB persistent vector storage
✅ Strict context-only answering
✅ CLI + Streamlit UI support
✅ Modular provider abstraction
✅ Chat history isolation (New chat ≠ old session)
✅ Duplicate chunk filtering
✅ Context relevance enforcement

🧠 RAG Pipeline Overview
User Query
     ↓
Semantic Retriever (ChromaDB)
     ↓
Relevant Legal Chunks
     ↓
LLM (Groq / HuggingFace / Hybrid)
     ↓
Answer + Verifiable Sources


Design Goals:

Accuracy over creativity

Context enforcement

Safe prompt engineering

Production reliability

🏗️ Production-Grade Architecture
legalrag/
│
├── config/                # Configuration management
│   └── settings.py
│
├── data/                  # Raw legal documents
├── uploads/               # User uploaded docs
├── chroma_db/             # Persistent vector DB
│
├── src/
│   ├── ingestion/         # Document → Embeddings
│   ├── retrieval/         # Semantic search
│   ├── generation/        # RAG pipeline
│   ├── providers/         # LLM abstraction layer
│   ├── evaluation/        # Metrics
│   ├── ui/                # Streamlit frontend
│   └── utils/             # Helper functions
│
├── requirements.txt
├── .env
└── README.md


✔ Every folder contains __init__.py
✔ Clean modular separation
✔ Provider factory pattern
✔ Production-safe imports

🛠 Technology Stack
Component	Technology
Language	Python 3.10
RAG Framework	LangChain
Vector Database	ChromaDB (Persistent)
Embeddings	HuggingFace Sentence Transformers
LLM Providers	Groq / HuggingFace / Hybrid
UI	Streamlit
Configuration	Pydantic Settings
Logging	Loguru
⚙️ Installation Guide
1️⃣ Create Environment
conda create -n legalrag310 python=3.10
conda activate legalrag310

2️⃣ Install Dependencies
pip install -r requirements.txt

3️⃣ Configure Environment

Create .env file:

API_PROVIDER=groq
GROQ_API_KEY=your_key_here

CHROMA_PERSIST_DIRECTORY=./data/chroma_db
CHROMA_COLLECTION_NAME=legal_documents

📥 Ingest Legal Documents

Place .txt or .pdf files inside:

data/uploads/


Run ingestion:

python src/ingestion/run_ingestion.py


Verify vector storage:

python check_chroma.py

🔍 Ask Legal Questions (CLI)
python -m src.generation.rag_pipeline


Example:

Input:

Explain Section 58 of the Indian Evidence Act


Output:

ANSWER:
Facts admitted need not be proved. If parties admit a fact in writing or in court, no further proof is required.

SOURCES:
Evidence_Act_1872_p29_c0.txt | chunk 3

🖥️ Run Streamlit UI
streamlit run app.py


Features:

Fresh chat isolation

Section-based queries

Source traceability

Clean legal answer formatting

📊 Reliability & Safety

✔ Context-only enforcement
✔ Duplicate chunk filtering
✔ Strict refusal on missing content
✔ No external knowledge injection
✔ Controlled temperature for deterministic output

🚀 Deployment Ready

Supports:

Streamlit Cloud

HuggingFace Spaces

Local production deployment

Modular provider switching

Persistent vector DB

🧑‍💼 Interview Value

This project demonstrates:

Real-world RAG architecture

Vector database engineering

Prompt safety & hallucination control

Modular Python system design

Legal-domain AI implementation

Multi-provider LLM abstraction

Production-level folder structure

🔮 Future Improvements

Section-aware retriever (Exact section matching)

Multi-Act filtering system

Citation highlighting in UI

Answer confidence scoring

PDF upload directly via UI

Legal summarization mode

📄 License

MIT License

🙌 Acknowledgements

LangChain

ChromaDB

HuggingFace

Groq

Indian Legal Open Data
