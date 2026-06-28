"""
Enterprise Knowledge Assistant — Main FastAPI Application Entry Point.

Initializes the application, sets up the RAG pipeline components,
and mounts the API routes and static frontend files.
"""

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from config import settings
import app.routes as routes
from app.bm25_index import BM25Index
from app.confidence import ConfidenceEngine
from app.conversation import ConversationMemory
from app.document_processor import DocumentProcessor
from app.embedding_engine import EmbeddingEngine
from app.guardrails import ContextGuardrails
from app.llm_engine import LLMEngine
from app.rag_pipeline import RAGPipeline
from app.retriever import HybridRetriever
from app.vector_store import VectorStore

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle events: startup and shutdown."""
    logger.info("Initializing Enterprise Knowledge Assistant...")

    # 1. Initialize core components
    doc_processor = DocumentProcessor(
        chunk_size=settings.chunk_size, 
        chunk_overlap=settings.chunk_overlap
    )
    embedding_engine = EmbeddingEngine(model_name=settings.embedding_model)
    vector_store = VectorStore(persist_directory=settings.chroma_db_path)
    bm25_index = BM25Index(index_path=settings.bm25_index_path)

    # 2. Build retriever & pipeline layers
    retriever = HybridRetriever(
        vector_store=vector_store,
        bm25_index=bm25_index,
        embedding_engine=embedding_engine,
        cross_encoder_model=settings.cross_encoder_model,
        top_k=settings.top_k_retrieval,
        top_n=settings.top_n_rerank,
    )
    
    # Preload models
    logger.info("Preloading embedding model...")
    _ = embedding_engine.model
    logger.info("Preloading cross-encoder...")
    from app.retriever import _get_cross_encoder
    _ = _get_cross_encoder(settings.cross_encoder_model)
    
    guardrails = ContextGuardrails(confidence_threshold=settings.confidence_threshold)
    confidence_engine = ConfidenceEngine(
        retrieval_weight=settings.retrieval_weight,
        reranker_weight=settings.reranker_weight,
    )
    llm_engine = LLMEngine(
        api_key=settings.google_api_key,
        gemini_model=settings.gemini_model,
        ollama_model=settings.ollama_model,
    )
    memory = ConversationMemory(max_turns=settings.max_conversation_turns)

    # 3. Assemble full RAG Pipeline
    pipeline = RAGPipeline(
        retriever=retriever,
        guardrails=guardrails,
        llm_engine=llm_engine,
        confidence_engine=confidence_engine,
        memory=memory,
    )

    # 4. Inject into routes
    routes.pipeline = pipeline
    routes.vector_store = vector_store
    routes.bm25_index = bm25_index
    routes.doc_processor = doc_processor
    routes.embedding_engine = embedding_engine

    # 5. Auto-ingest sample documents if vector store is empty
    if vector_store.count == 0 and os.path.exists(settings.documents_dir):
        logger.info(f"Vector store empty. Auto-ingesting documents from {settings.documents_dir}...")
        chunks = doc_processor.process_directory(settings.documents_dir)
        if chunks:
            texts = [c.text for c in chunks]
            embeddings = embedding_engine.embed_texts(texts)
            vector_store.add_chunks(chunks, embeddings)
            bm25_index.add_chunks(chunks)
            logger.info("Auto-ingestion complete!")
        else:
            logger.info("No valid documents found for auto-ingestion.")

    logger.info("Enterprise Knowledge Assistant is ready!")
    yield
    
    logger.info("Shutting down...")


# Create FastAPI app
app = FastAPI(
    title="Enterprise Knowledge Assistant",
    description="RAG-powered API for answering questions from enterprise documents",
    version="1.0.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount API routes
app.include_router(routes.router, prefix="/api")

# Mount static frontend
static_dir = os.path.join(os.path.dirname(__file__), "static")
if not os.path.exists(static_dir):
    os.makedirs(static_dir)

app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app", 
        host=settings.app_host, 
        port=settings.app_port, 
        reload=settings.app_debug
    )
