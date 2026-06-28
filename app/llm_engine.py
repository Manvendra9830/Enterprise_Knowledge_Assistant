"""
LLM Engine — Interfaces with LLMs for query rewriting and answer generation.

Implements a strict 3-tier fallback strategy:
- Tier 1: Google Gemini (gemini-2.0-flash) - primary
- Tier 2: Ollama (local model, e.g., llama3.1:8b) - auto fallback
- Tier 3: Retrieval-Only mode - deterministic plain-text answer based on top chunks
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

from google import genai
from google.genai import types
from google.genai.errors import APIError, ClientError
from openai import OpenAI
import httpx

from app.models import RankedChunk

logger = logging.getLogger(__name__)


class LLMEngine:
    """Manages interactions with the LLM via a 3-tier fallback strategy."""

    def __init__(self, api_key: str, gemini_model: str = "gemini-2.0-flash", ollama_model: str = "llama3.1:8b"):
        if not api_key:
            logger.warning("Google API Key is missing. Gemini calls will fail and fallback immediately.")
        
        self.api_key = api_key
        self.gemini_model = gemini_model
        self.ollama_model = ollama_model
        
        # Initialize Gemini client
        self.gemini_client = genai.Client(api_key=api_key) if api_key else None
        
        # Initialize Ollama OpenAI-compatible client
        try:
            self.ollama_client = OpenAI(
                base_url='http://localhost:11434/v1',
                api_key='ollama', # required, but unused
            )
        except Exception as e:
            logger.error(f"Failed to initialize Ollama OpenAI client: {e}")
            self.ollama_client = None

    # ------------------------------------------------------------------
    # Query Rewriting
    # ------------------------------------------------------------------

    def rewrite_query(self, query: str, conversation_context: Optional[str] = None) -> str:
        """Rewrite a user query to make it standalone."""
        if not conversation_context:
            return query
            
        prompt = (
            "Rewrite the follow-up question to be standalone by replacing pronouns with the subject from the history.\n"
            "Example:\nHistory: Turn 1:\nUser: Who is Pihu?\nAssistant: Pihu is a developer.\nFollow-up: Where does she work?\nRewrite: Where does Pihu work?\n\n"
            f"History:\n{conversation_context}\n"
            f"Follow-up: {query}\n"
            "Rewrite:"
        )

        # Tier 1: Gemini
        try:
            if self.gemini_client:
                logger.debug(f"Rewriting query with Gemini: '{query}'")
                response = self.gemini_client.models.generate_content(
                    model=self.gemini_model,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=0.1,
                        max_output_tokens=100,
                    )
                )
                return self._clean_rewritten_query(response.text, query)
        except (APIError, ClientError) as e:
            logger.warning(f"Gemini API/Client Error during rewrite: {e}. Falling back to Ollama.")
        except Exception as e:
            logger.warning(f"Gemini rewrite failed ({type(e).__name__}: {e}). Falling back to Ollama.")

        # Tier 2: Ollama
        try:
            if self.ollama_client:
                logger.debug(f"Rewriting query with Ollama: '{query}'")
                response = self.ollama_client.chat.completions.create(
                    model=self.ollama_model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                    max_tokens=100
                )
                if response.choices and response.choices[0].message.content:
                    return self._clean_rewritten_query(response.choices[0].message.content, query)
        except Exception as e:
            logger.warning(f"Ollama rewrite failed ({type(e).__name__}: {e}). Falling back to original query.")

        # Tier 3: Original Query
        return query

    def _clean_rewritten_query(self, text: str, original: str) -> str:
        if not text:
            return original
        rewritten = text.strip().replace('"', '').replace('\n', ' ')
        # Guard against LLM hallucinating a massive essay instead of a query
        if len(rewritten) > len(original) + 150:
            logger.warning(f"Rewritten query too long. Discarding: {rewritten}")
            return original
            
        logger.info(f"Query rewritten: '{original}' -> '{rewritten}'")
        return rewritten

    # ------------------------------------------------------------------
    # Answer Generation
    # ------------------------------------------------------------------

    def generate_answer(
        self, 
        query: str, 
        chunks: list[RankedChunk], 
        conversation_context: Optional[str] = None
    ) -> Tuple[str, str]:
        """
        Generate an answer using a strict 3-tier fallback.
        Returns: (answer_text, source_identifier)
        """
        if not chunks:
            return "I don't have enough information in the uploaded documents to answer that question.", "retrieval"

        system_instruction, user_prompt = self._build_prompts(query, chunks, conversation_context)

        # ------------------------------------------------------------------
        # Tier 1: Gemini
        # ------------------------------------------------------------------
        logger.info("Attempting Gemini generation (Tier 1)...")
        try:
            if self.gemini_client:
                response = self.gemini_client.models.generate_content(
                    model=self.gemini_model,
                    contents=user_prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=system_instruction,
                        temperature=0.2,
                        max_output_tokens=1024,
                    )
                )
                if response.text:
                    logger.info("Gemini generation success.")
                    return response.text.strip(), "gemini"
            else:
                logger.info("Gemini client not initialized. Skipping Tier 1.")
        except (APIError, ClientError) as e:
            logger.error(f"Gemini API/Client Error (possible 429 quota exhausted or auth issue): {e}")
            logger.info("Fallback activated -> Trying Tier 2 (Ollama)")
        except Exception as e:
            logger.error(f"Gemini generation failed abruptly: {type(e).__name__} - {e}", exc_info=True)
            logger.info("Fallback activated -> Trying Tier 2 (Ollama)")

        # ------------------------------------------------------------------
        # Tier 2: Ollama
        # ------------------------------------------------------------------
        logger.info("Attempting Ollama generation (Tier 2)...")
        try:
            if self.ollama_client:
                response = self.ollama_client.chat.completions.create(
                    model=self.ollama_model,
                    messages=[
                        {"role": "system", "content": system_instruction},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.2,
                    max_tokens=1024
                )
                if response.choices and response.choices[0].message.content:
                    logger.info("Ollama generation success.")
                    return response.choices[0].message.content.strip(), "ollama"
            else:
                logger.info("Ollama client not initialized. Skipping Tier 2.")
        except Exception as e:
            logger.error(f"Ollama generation failed: {type(e).__name__} - {e}", exc_info=True)
            logger.info("Fallback activated -> Trying Tier 3 (Retrieval-Only)")

        # ------------------------------------------------------------------
        # Tier 3: Retrieval-Only Mode
        # ------------------------------------------------------------------
        return self._generate_retrieval_only(chunks), "retrieval"

    def _build_prompts(self, query: str, chunks: list[RankedChunk], conversation_context: Optional[str]) -> Tuple[str, str]:
        context_parts = []
        for i, c in enumerate(chunks, start=1):
            doc_ref = f"[Source: {c.chunk.document_name}, Page {c.chunk.page_number}]"
            context_parts.append(f"--- Document Chunk {i} {doc_ref} ---\n{c.chunk.text}\n")
            
        formatted_context = "\n".join(context_parts)

        system_instruction = (
            "You are a helpful, professional Enterprise Knowledge Assistant. "
            "Your task is to answer the user's question based STRICTLY on the provided Document Chunks. "
            "\n\nRules:"
            "\n1. ONLY use information from the provided chunks. Do NOT use outside knowledge."
            "\n2. If the answer is not contained in the chunks, say exactly: "
            "'I don't have enough information in the uploaded documents to answer that question.'"
            "\n3. Be concise, clear, and direct."
            "\n4. If you synthesize information from multiple chunks, ensure the final answer is coherent."
            "\n5. Use formatting (bullet points, bold text) where it improves readability."
        )

        user_prompt = f"Context Documents:\n\n{formatted_context}\n\nUser Question: {query}"
        if conversation_context:
             user_prompt = f"Previous Conversation:\n{conversation_context}\n\n" + user_prompt
             
        return system_instruction, user_prompt

    def _generate_retrieval_only(self, chunks: list[RankedChunk]) -> str:
        """Deterministic plain-text answer built from top chunks without an LLM."""
        logger.info("Using Retrieval-Only fallback mode (Tier 3)")
        
        answer = "**LLM services are currently unavailable. Below is the most relevant information retrieved from the knowledge base:**\n\n"
        
        # Take top 3 chunks maximum
        for i, c in enumerate(chunks[:3], 1):
            excerpt = c.chunk.text.strip().replace("\n", " ")
            if len(excerpt) > 300:
                excerpt = excerpt[:300] + "..."
            answer += f"**{i}. {c.chunk.document_name} (Page {c.chunk.page_number})**\n> {excerpt}\n\n"
            
        return answer
