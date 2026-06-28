"""
API Routes — FastAPI endpoint handlers.
"""

from __future__ import annotations

import logging
import os
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
    docs = 0
    chunks = 0
    
    # Vector store check
    chromadb_status = "healthy"
    try:
        if vector_store:
            doc_list = vector_store.get_all_documents()
            docs = len(doc_list)
            chunks = vector_store.count
        else:
            chromadb_status = "unavailable"
    except Exception as e:
        logger.error(f"Health check failed to query vector store: {e}")
        chromadb_status = "error"

    # Component checks
    embedding_status = "loaded" if embedding_engine else "unavailable"
    reranker_status = "loaded" if pipeline and pipeline.retriever else "unavailable"
    
    # -------------------------------------------------------------
    # Gemini Check (Actual Generation)
    # -------------------------------------------------------------
    gemini_status = "unavailable"
    if pipeline and pipeline.llm_engine and pipeline.llm_engine.gemini_client:
        try:
            from google.genai import types
            response = pipeline.llm_engine.gemini_client.models.generate_content(
                model=settings.gemini_model,
                contents="health check",
                config=types.GenerateContentConfig(max_output_tokens=5)
            )
            if response.text:
                gemini_status = "available"
        except Exception as e:
            logger.error(f"Gemini health generation failed: {type(e).__name__} - {e}")

    # -------------------------------------------------------------
    # Ollama Check (Actual Generation)
    # -------------------------------------------------------------
    ollama_status = "unavailable"
    if pipeline and pipeline.llm_engine and pipeline.llm_engine.ollama_client:
        try:
            response = pipeline.llm_engine.ollama_client.chat.completions.create(
                model=settings.ollama_model,
                messages=[{"role": "user", "content": "health check"}],
                max_tokens=5,
                timeout=10.0
            )
            if response.choices and response.choices[0].message.content:
                ollama_status = "available"
        except Exception as e:
            logger.error(f"Ollama health generation failed: {type(e).__name__} - {e}")

    fallback_mode = (gemini_status != "available")

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
        response = pipeline.process_query(request)
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
        content = await file.read()
        logger.info(f"Received upload: {file.filename}")
        
        # Process document
        chunks = doc_processor.process_file(file.filename, content)
        if not chunks:
            raise HTTPException(status_code=400, detail="Could not extract text from document")
            
        # Generate embeddings
        texts = [c.text for c in chunks]
        embeddings = embedding_engine.embed_texts(texts)
        
        # Add to Vector Store
        vector_store.add_chunks(chunks, embeddings)
        
        # Add to BM25 Index
        bm25_index.add_chunks(chunks)
        
        # Save physical file
        doc_path = os.path.join(settings.documents_dir, file.filename)
        os.makedirs(settings.documents_dir, exist_ok=True)
        with open(doc_path, "wb") as f:
            f.write(content)
        
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
