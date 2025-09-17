"""AI Conversation Memory implementation for novelWriter AI Copilot."""

from __future__ import annotations

# import json  # Lint disable: unused until persistence arrives
import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import uuid4

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ConversationTurn:
    """Represents a single turn in an AI conversation."""
    
    turn_id: str = field(default_factory=lambda: uuid4().hex)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    user_input: str = ""
    ai_response: str = ""
    context_scope: str = "current_document"
    context_summary: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize conversation turn to dictionary."""
        return {
            "turn_id": self.turn_id,
            "timestamp": self.timestamp.isoformat(),
            "user_input": self.user_input,
            "ai_response": self.ai_response,
            "context_scope": self.context_scope,
            "context_summary": self.context_summary,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConversationTurn":
        """Deserialize conversation turn from dictionary."""
        return cls(
            turn_id=data.get("turn_id", uuid4().hex),
            timestamp=datetime.fromisoformat(data.get("timestamp", datetime.now(timezone.utc).isoformat())),
            user_input=data.get("user_input", ""),
            ai_response=data.get("ai_response", ""),
            context_scope=data.get("context_scope", "current_document"),
            context_summary=data.get("context_summary", ""),
            metadata=data.get("metadata", {})
        )


class ConversationMemory:
    """Manages conversation history and memory for AI Copilot sessions.
    
    Provides basic session memory capabilities including:
    - Storing recent conversation turns
    - Retrieving relevant conversation history
    - Context-aware conversation filtering
    - Memory persistence (optional)
    """
    
    def __init__(self, max_turns: int = 20) -> None:
        """Initialize conversation memory.
        
        Args:
            max_turns: Maximum number of conversation turns to keep in memory
        """
        self._max_turns = max_turns
        self._conversations: deque[ConversationTurn] = deque(maxlen=max_turns)
        self._session_id = uuid4().hex
        self._created_at = datetime.now(timezone.utc)
        
        logger.debug("ConversationMemory initialized with max_turns=%d, session_id=%s", 
                    max_turns, self._session_id)
    
    @property
    def session_id(self) -> str:
        """Get the current session identifier."""
        return self._session_id
    
    @property
    def created_at(self) -> datetime:
        """Get session creation timestamp."""
        return self._created_at
    
    @property
    def turn_count(self) -> int:
        """Get the number of conversation turns in memory."""
        return len(self._conversations)
    
    def add_turn(
        self,
        user_input: str,
        ai_response: str,
        context_scope: str = "current_document",
        context_summary: str = "",
        metadata: Optional[Dict[str, Any]] = None
    ) -> ConversationTurn:
        """Add a new conversation turn to memory.
        
        Args:
            user_input: The user's input/question
            ai_response: The AI's response
            context_scope: The context scope used for this turn
            context_summary: Brief summary of context used
            metadata: Additional metadata for the turn
            
        Returns:
            The created ConversationTurn object
        """
        turn = ConversationTurn(
            user_input=user_input.strip(),
            ai_response=ai_response.strip(),
            context_scope=context_scope,
            context_summary=context_summary,
            metadata=metadata or {}
        )
        
        self._conversations.append(turn)
        
        logger.debug("Added conversation turn %s (scope: %s, session: %s)", 
                    turn.turn_id, context_scope, self._session_id)
        
        return turn
    
    def get_recent_turns(self, count: int = 5) -> List[ConversationTurn]:
        """Get the most recent conversation turns.
        
        Args:
            count: Number of recent turns to retrieve
            
        Returns:
            List of recent conversation turns, most recent first
        """
        recent = list(self._conversations)[-count:] if count > 0 else list(self._conversations)
        return list(reversed(recent))  # Most recent first
    
    def get_turns_by_scope(self, scope: str, count: int = 10) -> List[ConversationTurn]:
        """Get conversation turns filtered by context scope.
        
        Args:
            scope: Context scope to filter by
            count: Maximum number of turns to return
            
        Returns:
            List of conversation turns matching the scope, most recent first
        """
        matching_turns = [turn for turn in self._conversations if turn.context_scope == scope]
        recent_matching = matching_turns[-count:] if count > 0 else matching_turns
        return list(reversed(recent_matching))  # Most recent first
    
    def get_relevant_context(
        self,
        current_scope: str,
        max_turns: int = 3,
        include_cross_scope: bool = True
    ) -> List[ConversationTurn]:
        """Get conversation turns relevant to the current context.
        
        Args:
            current_scope: Current context scope
            max_turns: Maximum number of turns to return
            include_cross_scope: Whether to include turns from other scopes
            
        Returns:
            List of relevant conversation turns, most recent first
        """
        relevant_turns = []
        
        # First priority: turns with same scope
        same_scope_turns = self.get_turns_by_scope(current_scope, max_turns)
        relevant_turns.extend(same_scope_turns)
        
        # Second priority: recent turns from other scopes (if space allows and enabled)
        if include_cross_scope and len(relevant_turns) < max_turns:
            remaining_slots = max_turns - len(relevant_turns)
            recent_turns = self.get_recent_turns(remaining_slots * 2)  # Get more to filter
            
            # Add turns from other scopes that aren't already included
            included_ids = {turn.turn_id for turn in relevant_turns}
            for turn in recent_turns:
                if turn.turn_id not in included_ids and turn.context_scope != current_scope:
                    relevant_turns.append(turn)
                    if len(relevant_turns) >= max_turns:
                        break
        
        # Sort by timestamp, most recent first
        relevant_turns.sort(key=lambda t: t.timestamp, reverse=True)
        return relevant_turns[:max_turns]
    
    def search_turns(self, query: str, max_results: int = 5) -> List[ConversationTurn]:
        """Search conversation turns by content.
        
        Args:
            query: Search query string
            max_results: Maximum number of results to return
            
        Returns:
            List of conversation turns matching the query, most recent first
        """
        if not query.strip():
            return []
        
        query_lower = query.lower().strip()
        matching_turns = []
        
        for turn in self._conversations:
            # Search in user input and AI response
            if (query_lower in turn.user_input.lower() or 
                query_lower in turn.ai_response.lower() or
                query_lower in turn.context_summary.lower()):
                matching_turns.append(turn)
        
        # Sort by timestamp, most recent first
        matching_turns.sort(key=lambda t: t.timestamp, reverse=True)
        return matching_turns[:max_results]
    
    def clear_memory(self) -> None:
        """Clear all conversation memory and start a new session."""
        old_session_id = self._session_id
        self._conversations.clear()
        self._session_id = uuid4().hex
        self._created_at = datetime.now(timezone.utc)
        
        logger.info("Cleared conversation memory. Old session: %s, New session: %s", 
                   old_session_id, self._session_id)
    
    def get_memory_summary(self) -> Dict[str, Any]:
        """Get a summary of current memory state.
        
        Returns:
            Dictionary containing memory statistics and metadata
        """
        scope_counts = {}
        for turn in self._conversations:
            scope = turn.context_scope
            scope_counts[scope] = scope_counts.get(scope, 0) + 1
        
        return {
            "session_id": self._session_id,
            "created_at": self._created_at.isoformat(),
            "total_turns": len(self._conversations),
            "max_turns": self._max_turns,
            "scope_distribution": scope_counts,
            "oldest_turn": self._conversations[0].timestamp.isoformat() if self._conversations else None,
            "newest_turn": self._conversations[-1].timestamp.isoformat() if self._conversations else None
        }
    
    def export_conversation(self) -> Dict[str, Any]:
        """Export entire conversation history as JSON-serializable data.
        
        Returns:
            Dictionary containing all conversation data
        """
        return {
            "session_metadata": {
                "session_id": self._session_id,
                "created_at": self._created_at.isoformat(),
                "max_turns": self._max_turns,
                "export_timestamp": datetime.now(timezone.utc).isoformat()
            },
            "conversations": [turn.to_dict() for turn in self._conversations]
        }
    
    def import_conversation(self, data: Dict[str, Any]) -> bool:
        """Import conversation history from exported data.
        
        Args:
            data: Previously exported conversation data
            
        Returns:
            True if import was successful, False otherwise
        """
        try:
            # Validate data structure
            if "conversations" not in data:
                logger.error("Invalid conversation data: missing 'conversations' key")
                return False
            
            # Clear current memory
            self._conversations.clear()
            
            # Import conversation turns
            for turn_data in data["conversations"]:
                try:
                    turn = ConversationTurn.from_dict(turn_data)
                    self._conversations.append(turn)
                except Exception as exc:
                    logger.warning("Failed to import conversation turn: %s", exc)
                    continue
            
            # Update session metadata if available
            if "session_metadata" in data:
                metadata = data["session_metadata"]
                if "session_id" in metadata:
                    self._session_id = metadata["session_id"]
                if "created_at" in metadata:
                    self._created_at = datetime.fromisoformat(metadata["created_at"])
            
            logger.info("Successfully imported %d conversation turns", len(self._conversations))
            return True
            
        except Exception as exc:
            logger.error("Failed to import conversation data: %s", exc)
            return False