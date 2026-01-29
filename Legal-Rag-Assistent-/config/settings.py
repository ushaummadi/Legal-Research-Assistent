import os
from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file='.env',
        env_file_encoding='utf-8',
        extra='ignore',
        case_sensitive=False
    )

    # Provider
    api_provider: str = Field(default="gemini")

    # Groq
    groq_api_key: Optional[str] = Field(default="")
    groq_llm_model: str = Field(default="llama-3.1-8b-instant")

    # HuggingFace
    huggingface_api_key: Optional[str] = Field(default="")
    hf_embedding_model: str = Field(default="sentence-transformers/all-MiniLM-L6-v2")
    hf_llm_model: str = Field(default="HuggingFaceH4/zephyr-7b-beta")
    
    # ChromaDB
    DOCS_DIR: str = Field(default="./data")  # PDFs here: iea_1872.pdf
    chroma_persist_directory: str = Field(default="./data/chroma_db")
    chroma_collection_name: str = Field(default="legal_documents")
    
    # Data
    uploads_dir: str = Field(default="./data/uploads")
    DOCS_DIR: str = Field(default="./data/documents")  # ← ADDED
    TOP_K: int = Field(default=5)  # ← ADDED
    
    # Chunking
    chunk_size: int = Field(default=600)
    chunk_overlap: int = Field(default=100)
    
    # LLM
    temperature: float = Field(default=0.2)
    max_output_tokens: int = Field(default=1200)

settings = Settings()

# Create directories
os.makedirs(settings.uploads_dir, exist_ok=True)
os.makedirs(settings.chroma_persist_directory, exist_ok=True)
