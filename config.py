"""
Enterprise Knowledge Assistant — Centralized Configuration

All application settings are managed here, loaded from environment variables
with sensible defaults. Uses pydantic-settings for validation.
"""

import os
from pathlib import Path
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

load_dotenv()

# Project root directory
BASE_DIR = Path(__file__).resolve().parent


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # --- API Keys ---
    google_api_key: str = ""

    # --- Application ---
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    app_debug: bool = False

    # --- RAG Configuration ---
    chunk_size: int = 500
    chunk_overlap: int = 50
    top_k_retrieval: int = 10
    top_n_rerank: int = 5
    confidence_threshold: float = 0.35

    # --- Model Configuration ---
    embedding_model: str = "all-MiniLM-L6-v2"
    cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    gemini_model: str = "gemini-2.0-flash"
    ollama_model: str = "qwen3:8b"

    # --- Storage Paths ---
    chroma_db_path: str = str(BASE_DIR / "chroma_db")
    bm25_index_path: str = str(BASE_DIR / "bm25_index.pkl")
    documents_dir: str = str(BASE_DIR / "documents")

    # --- Conversation Memory ---
    max_conversation_turns: int = 5

    # --- Confidence Scoring Weights ---
    retrieval_weight: float = 0.6
    reranker_weight: float = 0.4

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


# Singleton instance
settings = Settings()
