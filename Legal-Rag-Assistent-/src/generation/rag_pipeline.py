from loguru import logger
import re
from typing import List, Dict, Any
from langchain_core.documents import Document
from src.retrieval.retriever import get_retriever
from src.providers.factory import ProviderFactory

# 🔥 Global chat history (persists across queries)
chat_history_store: List[Dict[str, str]] = []

SYSTEM_RULES = """You are LegalGPT - Indian Evidence Act expert.
⚠️ STRICT RULES:
1. Answer ONLY using the provided CONTEXT.
2. If the context provided below is empty or irrelevant, you MUST say exactly:
   "**No relevant sections found in indexed documents.**"
3. Do NOT make up answers. Do NOT use outside knowledge.
4. Do NOT mention source filenames or say "Based on the provided context" - just give the answer directly.

Format:
**Section X**: [Direct Answer/Quote]
"""

def format_context(docs: List[Document]) -> str:
    """Format with score filtering and clean labels"""
    parts = []
    relevant_count = 0  # ✅ Fixed NameError by initializing here
    
    for d in docs:
        meta = d.metadata
        score = meta.get("score", 0)
        
        # 🛑 SCORE FILTER: Ignore weak matches
        if score < 4.0:
            continue
            
        relevant_count += 1
        
        # Extract metadata safely
        section_info = meta.get("section", "N/A")
        
        # ✅ Fixed NameError by defining display_label inside the loop
        display_label = f"Section {section_info}" if section_info != "N/A" else "Legal Document"
        
        # Append without filename to keep context clean
        parts.append(f"[{display_label} | score:{score}] {d.page_content}")
    
    if relevant_count == 0:
        return ""  # Return empty if nothing is good enough
        
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

    # Retrieve relevant documents
    docs = retriever.get_relevant_documents(question)
    
    # Apply filtering inside format_context
    context = format_context(docs)
    
    # 🛑 If context is empty after filtering, stop here.
    if not context.strip():
        fake_answer = "**No relevant sections found in indexed documents.**"
        chat_history_store.append({"role": "user", "content": question})
        chat_history_store.append({"role": "assistant", "content": fake_answer})
        return {
            "answer": fake_answer,
            "sources": [],
            "doc_count": 0
        }

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
    
    # Clean up any residual LLM filler like "Based on the context..."
    answer_text = re.sub(r"^(Based on the provided context|According to the documents),?\s*", "", answer_text, flags=re.IGNORECASE)
    
    chat_history_store.append({"role": "user", "content": question})
    chat_history_store.append({"role": "assistant", "content": answer_text})
    
    return {
        "answer": answer_text,
        "sources": [{"source": d.metadata.get("source"), "score": d.metadata.get("score", 0)} for d in docs if d.metadata.get("score", 0) >= 4.0],
        "doc_count": len(docs),
        "chat_history_len": len(chat_history_store) 
    }
