"""
LLM Engine — Interfaces with LLMs for query rewriting and answer generation.

Implements a strict 3-tier fallback strategy:
- Tier 1: Google Gemini (gemini-2.0-flash) - primary
- Tier 2: Ollama (local model, e.g., llama3.1:8b) - auto fallback
- Tier 3: Retrieval-Only mode - deterministic plain-text answer based on top chunks
"""

from __future__ import annotations

import logging
import re
import time
from typing import Optional, Tuple

from google import genai
from google.genai import types
from google.genai.errors import APIError, ClientError
import httpx
from openai import OpenAI

from app.models import RankedChunk

logger = logging.getLogger(__name__)


class LLMEngine:
    """Manages interactions with the LLM via a 3-tier fallback strategy."""

    def __init__(self, api_key: str, gemini_model: str = "gemini-2.0-flash", ollama_model: str = "llama3.1:8b"):
        api_key = (api_key or "").strip()
        if not api_key:
            logger.warning("Google API Key is missing. Gemini calls will fail and fallback immediately.")
        
        self.api_key = api_key
        self.gemini_model = gemini_model
        self.ollama_model = ollama_model
        self.ollama_base_url = "http://localhost:11434"
        self.gemini_last_error: Optional[str] = None
        self.gemini_disabled_until = 0.0
        
        # Initialize Gemini client
        self.gemini_client = genai.Client(api_key=api_key) if api_key else None
        
        # Initialize Ollama OpenAI-compatible client
        try:
            self.ollama_client = OpenAI(
                base_url=f'{self.ollama_base_url}/v1',
                api_key='ollama', # required, but unused
                timeout=12.0,
                max_retries=0,
            )
        except Exception as e:
            logger.error(f"Failed to initialize Ollama OpenAI client: {e}")
            self.ollama_client = None

    # ------------------------------------------------------------------
    # Provider status / failure handling
    # ------------------------------------------------------------------

    def gemini_status(self) -> str:
        """Return a non-generative Gemini status for health reporting."""
        if not self.api_key or not self.gemini_client:
            return "missing_api_key"
        if self._gemini_in_cooldown():
            return "quota_exhausted" if self.gemini_last_error == "quota_exhausted" else "cooldown"
        return "configured"

    def _gemini_in_cooldown(self) -> bool:
        return time.monotonic() < self.gemini_disabled_until

    def _handle_gemini_exception(self, exc: Exception, operation: str) -> None:
        """Classify Gemini failures and activate fail-fast cooldown where safe."""
        message = str(exc)
        status_code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
        lowered = message.lower()

        if status_code == 429 or "resource_exhausted" in lowered or "quota" in lowered:
            self.gemini_last_error = "quota_exhausted"
            retry_match = re.search(r"retryDelay': '(\d+)s", message) or re.search(r"retry in ([\d.]+)s", message)
            retry_seconds = float(retry_match.group(1)) if retry_match else 60.0
            self.gemini_disabled_until = time.monotonic() + max(30.0, min(retry_seconds, 300.0))
            logger.error(
                "Gemini %s failed: quota exhausted. Cooling down Gemini for %.0fs.",
                operation,
                self.gemini_disabled_until - time.monotonic(),
                exc_info=True,
            )
            return

        if status_code in {401, 403} or "api key" in lowered or "authentication" in lowered:
            self.gemini_last_error = "authentication_failed"
            self.gemini_disabled_until = time.monotonic() + 300.0
        elif "timeout" in lowered:
            self.gemini_last_error = "timeout"
            self.gemini_disabled_until = time.monotonic() + 30.0
        else:
            self.gemini_last_error = type(exc).__name__
            self.gemini_disabled_until = time.monotonic() + 30.0

        logger.error("Gemini %s failed: %s - %s", operation, type(exc).__name__, exc, exc_info=True)

    def _should_rewrite_query(self, query: str, conversation_context: Optional[str]) -> bool:
        """Avoid slow LLM rewrite for standalone follow-up turns."""
        if not conversation_context:
            return False
        q = query.lower().strip()
        followup_markers = {
            "it", "its", "they", "them", "their", "he", "him", "his", "she", "her",
            "this", "that", "these", "those", "there", "where", "what about", "how about",
        }
        tokens = set(re.findall(r"\b[a-z]+\b", q))
        return len(q) < 120 and (tokens.intersection(followup_markers) or "what about" in q or "how about" in q)

    def _heuristic_rewrite_query(self, query: str, conversation_context: Optional[str]) -> Optional[str]:
        """Fast path for simple pronoun follow-ups such as 'Where does she study?'."""
        if not conversation_context:
            return None

        candidates = re.findall(r"\b[A-Z][a-z]{2,}\b", conversation_context)
        ignored = {"User", "Assistant", "Turn", "Context", "Previous", "Document", "Source"}
        entities = [c for c in candidates if c not in ignored]
        if not entities:
            return None

        entity_counts = {entity: entities.count(entity) for entity in set(entities)}
        entity = sorted(entity_counts, key=lambda e: (-entity_counts[e], entities.index(e)))[0]
        replacements = {
            r"\bshe\b": entity,
            r"\bhe\b": entity,
            r"\bher\b": entity,
            r"\bhim\b": entity,
            r"\bthey\b": entity,
            r"\bthem\b": entity,
            r"\btheir\b": f"{entity}'s",
        }
        rewritten = query
        for pattern, value in replacements.items():
            rewritten = re.sub(pattern, value, rewritten, flags=re.IGNORECASE)

        return rewritten if rewritten != query else None

    # ------------------------------------------------------------------
    # Query Rewriting
    # ------------------------------------------------------------------

    def rewrite_query(self, query: str, conversation_context: Optional[str] = None) -> str:
        """Rewrite a user query to make it standalone."""
        if not self._should_rewrite_query(query, conversation_context):
            return query

        heuristic = self._heuristic_rewrite_query(query, conversation_context)
        if heuristic:
            logger.info(f"Query rewritten heuristically: '{query}' -> '{heuristic}'")
            return heuristic
            
        prompt = (
            "Rewrite the follow-up question to be standalone by replacing pronouns with the subject from the history.\n"
            "Example:\nHistory: Turn 1:\nUser: Who is Pihu?\nAssistant: Pihu is a developer.\nFollow-up: Where does she work?\nRewrite: Where does Pihu work?\n\n"
            f"History:\n{conversation_context}\n"
            f"Follow-up: {query}\n"
            "Rewrite:"
        )

        # Tier 1: Gemini
        try:
            if self.gemini_client and not self._gemini_in_cooldown():
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
            self._handle_gemini_exception(e, "rewrite")
        except Exception as e:
            self._handle_gemini_exception(e, "rewrite")

        # Tier 2: Ollama
        try:
            if self.ollama_client:
                logger.debug(f"Rewriting query with Ollama: '{query}'")
                rewritten = self._rewrite_with_ollama(prompt)
                if rewritten:
                    return self._clean_rewritten_query(rewritten, query)
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
            if self.gemini_client and not self._gemini_in_cooldown():
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
                    self.gemini_last_error = None
                    return response.text.strip(), "gemini"
            elif self.gemini_client:
                logger.info("Gemini is in cooldown after prior failure. Skipping Tier 1.")
            else:
                logger.info("Gemini client not initialized. Skipping Tier 1.")
        except (APIError, ClientError) as e:
            self._handle_gemini_exception(e, "generation")
            logger.info("Fallback activated -> Trying Tier 2 (Ollama)")
        except Exception as e:
            self._handle_gemini_exception(e, "generation")
            logger.info("Fallback activated -> Trying Tier 2 (Ollama)")

        # ------------------------------------------------------------------
        # Tier 2: Ollama
        # ------------------------------------------------------------------
        logger.info("Attempting Ollama generation (Tier 2)...")
        try:
            if self.ollama_client:
                content = self._generate_with_ollama(system_instruction, user_prompt)
                if content:
                    logger.info("Ollama generation success.")
                    return content.strip(), "ollama"
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

    def _generate_with_ollama(self, system_instruction: str, user_prompt: str) -> str:
        """
        Generate through native Ollama chat.

        qwen3:8b exposes thinking behavior that can return empty `content` via the
        OpenAI-compatible endpoint. The native API supports `think=False`, preserving
        the same Ollama/qwen3 fallback while returning usable content faster.
        """
        payload = {
            "model": self.ollama_model,
            "messages": [
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
            "think": False,
            "keep_alive": "30m",
            "options": {
                "temperature": 0.2,
                "num_predict": 256,
                "num_ctx": 4096,
            },
        }
        response = httpx.post(
            f"{self.ollama_base_url}/api/chat",
            json=payload,
            timeout=8.0,
        )
        response.raise_for_status()
        data = response.json()
        return (data.get("message") or {}).get("content") or ""

    def _rewrite_with_ollama(self, prompt: str) -> str:
        payload = {
            "model": self.ollama_model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "think": False,
            "keep_alive": "30m",
            "options": {
                "temperature": 0.1,
                "num_predict": 80,
                "num_ctx": 2048,
            },
        }
        response = httpx.post(
            f"{self.ollama_base_url}/api/chat",
            json=payload,
            timeout=5.0,
        )
        response.raise_for_status()
        data = response.json()
        return (data.get("message") or {}).get("content") or ""

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
