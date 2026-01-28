import os
from langchain_groq import ChatGroq
from config.settings import settings

try:
    import streamlit as st
except Exception:
    st = None


class GroqProvider:
    def embeddings(self):
        raise NotImplementedError("Use HF embeddings via HybridProvider (see below).")

    def llm(self):
        # Priority: env var -> Streamlit secrets -> settings (.env)
        key = os.getenv("GROQ_API_KEY")

        if not key and st is not None:
            # Works if your Cloud secrets is: GROQ_API_KEY = "..."
            key = st.secrets.get("GROQ_API_KEY", "")

            # Optional: if you used [grok] table
            if not key and "grok" in st.secrets:
                key = st.secrets["grok"].get("GROQ_API_KEY", "")

        if not key:
            key = getattr(settings, "groq_api_key", "")

        if not key:
            raise ValueError("GROQ_API_KEY missing. Add it in Streamlit Cloud → Settings → Secrets.")

        return ChatGroq(
            groq_api_key=key,
            model=getattr(settings, "groq_llm_model", "llama3-8b-8192"),
            temperature=getattr(settings, "temperature", 0.2),
            max_tokens=getattr(settings, "max_output_tokens", 1024),
        )
