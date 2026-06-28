"""
API Routes — FastAPI endpoint handlers.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Path
import httpx

from app.models import (
    AskRequest,
    AskResponse,
    DocumentInfo,
    FeedbackRequest,
    FeedbackResponse,
    HealthResponse,
)
from app.rag_pipeline import RAGPipeline
from app.vector_store import VectorStore
from app.bm25_index import BM25Index
from app.document_processor import DocumentProcessor
from app.embedding_engine import EmbeddingEngine
from app.profiling import log_latency_breakdown, timed_step
from config import settings

logger = logging.getLogger(__name__)

# Create the router
router = APIRouter()

# Global dependencies (injected in main.py)
pipeline: Optional[RAGPipeline] = None
vector_store: Optional[VectorStore] = None
bm25_index: Optional[BM25Index] = None
doc_processor: Optional[DocumentProcessor] = None
embedding_engine: Optional[EmbeddingEngine] = None


@router.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """System health check and status of all models."""
    timings: dict[str, float] = {}
    docs = 0
    chunks = 0
    
    # Vector store check
    chromadb_status = "healthy"
    with timed_step(timings, "chromadb"):
        try:
            if vector_store:
                doc_list = vector_store.get_all_documents()
                docs = len(doc_list)
                chunks = vector_store.count
            else:
                chromadb_status = "unavailable"
        except Exception as e:
            logger.error(f"Health check failed to query vector store: {e}", exc_info=True)
            chromadb_status = "error"

    # Component checks
    embedding_status = "loaded" if embedding_engine else "unavailable"
    reranker_status = "loaded" if pipeline and pipeline.retriever else "unavailable"
    
    # Gemini health must not spend quota or perform generation.
    gemini_status = "unavailable"
    if pipeline and pipeline.llm_engine:
        gemini_status = pipeline.llm_engine.gemini_status()

    # Ollama health uses the lightweight tags endpoint; no generation call.
    ollama_status = "unavailable"
    if pipeline and pipeline.llm_engine:
        try:
            with timed_step(timings, "ollama"):
                response = httpx.get("http://localhost:11434/api/tags", timeout=1.0)
            if response.status_code == 200:
                models = response.json().get("models", [])
                model_names = {m.get("name") or m.get("model") for m in models}
                ollama_status = "available" if settings.ollama_model in model_names else "model_missing"
            else:
                ollama_status = f"error_{response.status_code}"
        except Exception as e:
            logger.warning(f"Ollama health check failed: {type(e).__name__} - {e}")

    fallback_mode = gemini_status not in {"configured", "available"}
    log_latency_breakdown("health", timings)

    return HealthResponse(
        documents_loaded=docs,
        total_chunks=chunks,
        api="healthy",
        chromadb=chromadb_status,
        embedding_model=embedding_status,
        reranker=reranker_status,
        gemini=gemini_status,
        ollama=ollama_status,
        fallback_mode=fallback_mode,
    )


@router.post("/ask", response_model=AskResponse, tags=["Knowledge"])
async def ask_question(request: AskRequest):
    """
    Ask a question to the Enterprise Knowledge Assistant.
    Provides answer, cited sources with excerpts, and confidence score.
    """
    if not pipeline:
        raise HTTPException(status_code=503, detail="RAG Pipeline not initialized")
        
    try:
        start = time.perf_counter()
        response = pipeline.process_query(request)
        logger.info("request_latency endpoint=ask total=%.1fms", (time.perf_counter() - start) * 1000)
        return response
    except Exception as e:
        logger.exception("Error processing query")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/documents", response_model=list[DocumentInfo], tags=["Documents"])
async def list_documents():
    """List all ingested documents and their metadata."""
    if not vector_store:
        raise HTTPException(status_code=503, detail="Vector Store not initialized")
        
    try:
        docs = vector_store.get_all_documents()
        return [
            DocumentInfo(
                name=d["name"],
                file_type=d["file_type"],
                chunk_count=d["chunk_count"],
            ) for d in docs
        ]
    except Exception as e:
        logger.error(f"Error listing documents: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve documents")


@router.post("/upload", tags=["Documents"])
async def upload_document(file: UploadFile = File(...)):
    """
    Upload a document for ingestion.
    Extracts text, chunks, embeds, and adds to semantic/keyword indexes.
    """
    if not all([doc_processor, embedding_engine, vector_store, bm25_index]):
        raise HTTPException(status_code=503, detail="Ingestion components not initialized")
        
    try:
        timings: dict[str, float] = {}
        with timed_step(timings, "read_upload"):
            content = await file.read()
        logger.info(f"Received upload: {file.filename}")
        
        # Process document
        with timed_step(timings, "chunking"):
            chunks = doc_processor.process_file(file.filename, content)
        if not chunks:
            raise HTTPException(status_code=400, detail="Could not extract text from document")
            
        # Generate embeddings
        texts = [c.text for c in chunks]
        with timed_step(timings, "embedding"):
            embeddings = embedding_engine.embed_texts(texts)
        
        # Add to Vector Store
        with timed_step(timings, "chroma_insertion"):
            vector_store.add_chunks(chunks, embeddings)
        
        # Add to BM25 Index
        with timed_step(timings, "bm25_index_update"):
            bm25_index.add_chunks(chunks)
        
        # Save physical file
        with timed_step(timings, "file_save"):
            doc_path = os.path.join(settings.documents_dir, file.filename)
            os.makedirs(settings.documents_dir, exist_ok=True)
            with open(doc_path, "wb") as f:
                f.write(content)
        log_latency_breakdown("upload", timings)
        
        return {
            "status": "success", 
            "filename": file.filename, 
            "chunks_processed": len(chunks)
        }
    except Exception as e:
        logger.exception(f"Error uploading document: {file.filename}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/documents/{document_name}", tags=["Documents"])
async def delete_document(document_name: str = Path(...)):
    """
    Delete a document and rebuild indexes.
    """
    if not all([vector_store, bm25_index]):
        raise HTTPException(status_code=503, detail="Storage components not initialized")

    try:
        # 1. Remove from vector store
        deleted_vectors = vector_store.delete_document(document_name)
        
        # 2. Remove from BM25
        deleted_bm25 = bm25_index.remove_document(document_name)
        
        if deleted_vectors == 0 and deleted_bm25 == 0:
            raise HTTPException(status_code=404, detail="Document not found")
            
        # 3. Delete physical file
        doc_path = os.path.join(settings.documents_dir, document_name)
        if os.path.exists(doc_path):
            os.remove(doc_path)
        logger.info(f"Document removed: {document_name}")
            
        return {"status": "deleted", "document": document_name}
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error deleting document {document_name}: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete document")


@router.post("/feedback", response_model=FeedbackResponse, tags=["Analytics"])
async def submit_feedback(request: FeedbackRequest):
    """
    Submit user feedback for an answer (rating 1-5).
    In a real system, this would write to a database.
    """
    logger.info(f"Feedback received: Rating={request.rating}, Q='{request.question[:50]}'")
    return FeedbackResponse()
