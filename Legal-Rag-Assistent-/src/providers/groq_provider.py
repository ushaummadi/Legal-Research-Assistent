import os
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from config.settings import settings

try:
    import streamlit as st
except Exception:
    st = None


class GroqProvider:
    def embeddings(self):
        # ✅ Provide embeddings (so factory/pipeline won't crash)
        model_name = getattr(
            settings,
            "hf_embedding_model",
            "sentence-transformers/all-MiniLM-L6-v2",
        )
        return HuggingFaceEmbeddings(model_name=model_name)  # HF embeddings [web:223]

    def llm(self):
        # ✅ Read from env OR Streamlit Cloud Secrets OR .env settings
        key = os.getenv("GROQ_API_KEY")

        if not key and st is not None:
            # If secrets has: GROQ_API_KEY = "..."
            key = st.secrets.get("GROQ_API_KEY", "")

        if not key:
            key = getattr(settings, "groq_api_key", "")

        if not key:
            raise ValueError("GROQ_API_KEY missing. Add it in Streamlit Cloud → Secrets.")

        return ChatGroq(
            groq_api_key=key,
            model=getattr(settings, "groq_llm_model", "llama3-8b-8192"),
            temperature=getattr(settings, "temperature", 0.2),
            max_tokens=getattr(settings, "max_output_tokens", 1024),
        )
