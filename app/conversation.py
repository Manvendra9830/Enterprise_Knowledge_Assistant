"""
Conversation Memory — Manages session-based chat history.

Maintains context for follow-up questions, allowing the LLM Engine to
rewrite queries with proper pronoun resolution (e.g., "What is the policy?" ->
"How many days is it?" -> "How many days is the policy?").
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Optional

logger = logging.getLogger(__name__)


class ConversationMemory:
    """In-memory store for session-based chat histories."""

    def __init__(self, max_turns: int = 5):
        self.max_turns = max_turns
        # Dictionary mapping session_id -> list of (user_msg, system_msg)
        self._sessions: dict[str, list[tuple[str, str]]] = defaultdict(list)

    def add_turn(self, session_id: str, user_query: str, system_response: str):
        """Record a single conversation turn for a session."""
        if not session_id:
            return

        history = self._sessions[session_id]
        history.append((user_query, system_response))
        
        # Trim to max turns
        if len(history) > self.max_turns:
            self._sessions[session_id] = history[-self.max_turns:]
            
        logger.debug(f"Added turn to session {session_id}. Total turns: {len(self._sessions[session_id])}")

    def get_context_string(self, session_id: str) -> Optional[str]:
        """
        Format the recent conversation history into a string for the LLM.
        Returns None if no history exists.
        """
        if not session_id or session_id not in self._sessions:
            return None

        history = self._sessions[session_id]
        if not history:
            return None

        context_parts = []
        for i, (user_msg, system_msg) in enumerate(history, start=1):
            context_parts.append(f"Turn {i}:\nUser: {user_msg}\nAssistant: {system_msg}\n")

        return "\n".join(context_parts)

    def clear_session(self, session_id: str):
        """Clear the history for a specific session."""
        if session_id in self._sessions:
            del self._sessions[session_id]
            logger.info(f"Cleared session {session_id}")
