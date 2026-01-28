from loguru import logger
import re
from typing import List, Dict, Any
from langchain_core.documents import Document
from src.retrieval.retriever import get_retriever
from src.providers.factory import ProviderFactory

# 🔥 ZERO-IMPORT MEMORY (100% Works)
chat_history_store = []  

SYSTEM_RULES = """You are a legal research assistant.
CRITICAL RULES:
1. Answer ONLY using the EXACT text in CONTEXT below
2. If the context mentions the section number ANYWHERE (even in chunk metadata), use it
3. Section numbers are sequential - if context has 56,57 it's likely 58 is there too
4. Quote key phrases directly from context
5. If the context does NOT contain the answer: Simply say "Sorry,no rrsult found for above search."
5. NEVER say "not present" unless context has ZERO mention of that section
7. Use conversation history to maintain context

NEVER add external knowledge or summarize nearby sections unless specifically mentioned."""

def format_context(docs: list) -> str:
    parts = []
    for d in docs:
        src = d.metadata.get("source", "unknown")
        chunk = d.metadata.get("chunk", "?")
        parts.append(f"[{src} | chunk {chunk}]\n{d.page_content}")
    return "\n\n".join(parts)

def extract_section_number(question: str) -> str:
    match = re.search(r"section\s*(\d+)", question.lower())
    return match.group(1) if match else None

def normalize_question(question: str) -> str:
    section = extract_section_number(question)
    if section:
        return f"Explain Section {section} of the Indian Evidence Act, 1872"
    return question

def build_history_string(current_question: str) -> str:
    """Creates a history string without any LangChain memory objects."""
    if not chat_history_store:
        return f"CURRENT QUESTION: {current_question}"
    
    # Take last 3 exchanges
    history_str = "PREVIOUS CONVERSATION:\n"
    limit = 6 # 3 turns * 2 messages
    recent_history = chat_history_store[-limit:]
    
    for msg in recent_history:
        role = "User" if msg["role"] == "user" else "Assistant"
        content = msg["content"]
        # Truncate assistant answers to save tokens
        if role == "Assistant":
            content = content[:200] + "..." if len(content) > 200 else content
        history_str += f"{role}: {content}\n"
        
    return history_str + f"\nCURRENT QUESTION: {current_question}"

def answer_question(question: str) -> dict:
    retriever = get_retriever()
    provider = ProviderFactory.get_provider()
    llm = provider.llm()

    # ✅ USE ORIGINAL QUERY (No normalization!)
    docs = retriever.get_relevant_documents(question)
    
    # ✅ Always generate (never sorry!)
    if len(docs) == 0:
        docs = [Document(page_content="No exact chunks found. Try broader terms.", metadata={"source": "system"})]
    
    context = format_context(docs)
    
    full_prompt_input = build_history_string(question)
    prompt = f"""
{SYSTEM_RULES}

{full_prompt_input}

CONTEXT (Multiple Legal PDFs):
{context}

Precise Answer:"""
    
    resp = llm.invoke(prompt)
    answer_text = getattr(resp, "content", str(resp))
    
    chat_history_store.append({"role": "user", "content": question})
    chat_history_store.append({"role": "assistant", "content": answer_text})
    
    return {
        "answer": answer_text,
        "sources": [
            {"source": d.metadata.get("source", "unknown"), "chunk": d.metadata.get("chunk", "?")}
            for d in docs
        ],
    }
