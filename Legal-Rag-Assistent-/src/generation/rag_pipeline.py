from loguru import logger
import re
from typing import List, Dict, Any
from langchain_core.documents import Document
from src.retrieval.retriever import get_retriever
from src.providers.factory import ProviderFactory

# 🔥 Global chat history (persists across queries)
chat_history_store: List[Dict[str, str]] = []

SYSTEM_RULES = """You are LegalGPT - Indian Evidence Act expert.
CRITICAL:
1. Answer ONLY from CONTEXT (quote directly)
2. Mention section numbers from metadata/chunks
3. If partial match: "Based on available chunks..."
4. No results? "No relevant sections found in indexed docs."
5. Keep answers concise (200-400 words)
6. Use history for context continuity"""

def format_context(docs: List[Document]) -> str:
    """Format with source/chunk/score"""
    parts = []
    for d in docs:
        meta = d.metadata
        src = meta.get("source", "unknown")
        chunk = meta.get("chunk", "?")
        score = meta.get("score", 0)
        parts.append(f"[{src}:{chunk} | score:{score}] {d.page_content}")
    return "\n\n".join(parts)

def extract_section_number(question: str) -> str:
    match = re.search(r"section\s*(\d+)", question.lower())
    return match.group(1) if match else None

def build_history_string(question: str) -> str:
    if not chat_history_store:
        return f"CURRENT QUESTION: {question}"
    
    history_str = "RECENT CHAT:\n"
    recent = chat_history_store[-6:]  # Last 3 turns
    
    for msg in recent:
        role = "USER" if msg["role"] == "user" else "LEGALGPT"
        content = msg["content"][:150] + "..." if len(msg["content"]) > 150 else msg["content"]
        history_str += f"{role}: {content}\n"
    
    return history_str + f"\nCURRENT: {question}"

def answer_question(question: str) -> dict:
    retriever = get_retriever()
    provider = ProviderFactory.get_provider()
    llm = provider.llm()

    docs = retriever.get_relevant_documents(question)
    context = format_context(docs)
    
    prompt = f"""
{SYSTEM_RULES}

{build_history_string(question)}

CONTEXT:
{context}

ANSWER:"""
    
    try:
        resp = llm.invoke(prompt)
        answer_text = getattr(resp, "content", str(resp))
    except Exception as e:
        logger.error(f"LLM error: {e}")
        answer_text = "Error generating response. Check embeddings/LLM."
    
    chat_history_store.append({"role": "user", "content": question})
    chat_history_store.append({"role": "assistant", "content": answer_text})
    
    return {
        "answer": answer_text,
        "sources": [{"source": d.metadata.get("source"), "chunk": d.metadata.get("chunk"), "score": d.metadata.get("score", 0)} for d in docs],
        "doc_count": len(docs),
        "chat_history_len": len(chat_history_store) 
    }
