"""
RAG Pipeline — Orchestrates the complete query processing flow.

Pipeline steps:
1. Query Rewrite (using conversation history)
2. Hybrid Retrieval (Semantic + BM25 + RRF)
3. Cross-Encoder Re-ranking
4. Context Validation (Guardrails)
5. LLM Answer Generation
6. Confidence Scoring
7. Response Formatting (with Source Excerpts)
"""

from __future__ import annotations

import logging
from typing import Optional

from app.confidence import ConfidenceEngine
from app.conversation import ConversationMemory
from app.guardrails import ContextGuardrails
from app.llm_engine import LLMEngine
from app.models import AskRequest, AskResponse, SourceCitation
from app.profiling import log_latency_breakdown, timed_step
from app.retriever import HybridRetriever

logger = logging.getLogger(__name__)


class RAGPipeline:
    """Orchestrates the Enterprise Knowledge Assistant RAG flow."""

    def __init__(
        self,
        retriever: HybridRetriever,
        guardrails: ContextGuardrails,
        llm_engine: LLMEngine,
        confidence_engine: ConfidenceEngine,
        memory: ConversationMemory,
    ):
        self.retriever = retriever
        self.guardrails = guardrails
        self.llm_engine = llm_engine
        self.confidence_engine = confidence_engine
        self.memory = memory

    def process_query(self, request: AskRequest) -> AskResponse:
        """Process a user query and return a formulated response."""
        original_query = request.question.strip()
        session_id = request.session_id
        logger.info(f"Processing query: '{original_query}' (session: {session_id})")
        timings: dict[str, float] = {}

        # 1. Get conversation context
        with timed_step(timings, "memory"):
            context = self.memory.get_context_string(session_id) if session_id else None

        # 2. Query Rewriting
        search_query = original_query
        with timed_step(timings, "query_rewrite"):
            if context:
                search_query = self.llm_engine.rewrite_query(original_query, context)

        # 3. Hybrid Retrieval & Re-ranking
        with timed_step(timings, "retrieval"):
            ranked_chunks = self.retriever.retrieve(search_query)

        # 4. Context Validation / Guardrails
        with timed_step(timings, "guardrails"):
            validation = self.guardrails.validate_context(ranked_chunks, original_query)

        if not validation.proceed:
            # Blocked by guardrails -> return standard refusal, NO SOURCES
            logger.info("Guardrails rejected context. Returning standard refusal.")
            response_text = self.guardrails.REFUSAL_ANSWER
            
            # Still record turn in memory so it remembers the refusal
            if session_id:
                with timed_step(timings, "memory_update"):
                    self.memory.add_turn(session_id, original_query, response_text)
            log_latency_breakdown("ask", timings)
                
            return AskResponse(
                answer=response_text,
                sources=[],
                confidence=0.0,
                confidence_level=self.confidence_engine.get_confidence_level(0.0),
                session_id=session_id or "",
                answer_source="guardrails",
            )

        # 5. LLM Answer Generation
        # Only use top chunks that passed the guardrail
        top_chunks = ranked_chunks[:self.retriever.top_n]
        with timed_step(timings, "llm_generation"):
            answer, answer_source = self.llm_engine.generate_answer(search_query, top_chunks, context)

        # 6. Confidence Scoring
        with timed_step(timings, "confidence"):
            best_retrieval = max((c.retrieval_score for c in top_chunks), default=0.0)
            best_rerank = max((c.reranker_score for c in top_chunks), default=0.0)
            confidence = self.confidence_engine.calculate_confidence(best_retrieval, best_rerank)
            confidence_level = self.confidence_engine.get_confidence_level(confidence)

        # 7. Response Formatting (Source Citations with Excerpts)
        sources: list[SourceCitation] = []
        seen_docs = set()
        
        # If the LLM failed and we are in retrieval mode, we still show the sources.
        # But if the guardrails failed, we don't.
        with timed_step(timings, "citation_generation"):
            for c in top_chunks:
                doc_key = f"{c.chunk.document_name}::{c.chunk.page_number}"
                if doc_key not in seen_docs:
                    seen_docs.add(doc_key)
                    
                    text = c.chunk.text.strip()
                    excerpt = text[:150]
                    if len(text) > 150:
                        last_space = excerpt.rfind(" ")
                        if last_space > 100:
                            excerpt = excerpt[:last_space]
                        excerpt += "..."
                    
                    sources.append(
                        SourceCitation(
                            document=c.chunk.document_name,
                            page=c.chunk.page_number,
                            excerpt=excerpt,
                        )
                    )

        # Update Conversation Memory
        if session_id:
            with timed_step(timings, "memory_update"):
                self.memory.add_turn(session_id, original_query, answer)

        logger.info(f"Query processed successfully. Confidence: {confidence:.3f} ({confidence_level.value})")
        log_latency_breakdown("ask", timings)
        return AskResponse(
            answer=answer,
            sources=sources,
            confidence=confidence,
            confidence_level=confidence_level,
            session_id=session_id or "",
            answer_source=answer_source,
        )
