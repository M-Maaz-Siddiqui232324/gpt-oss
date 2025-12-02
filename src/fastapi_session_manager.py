"""FastAPI-native session management with in-memory storage"""
import uuid
import json
import logging
import os
from typing import Optional, Dict, List, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from collections import OrderedDict

logger = logging.getLogger(__name__)


@dataclass
class Message:
    """Represents a single message in the conversation"""
    role: str  # "user" or "assistant"
    content: str
    timestamp: str
    context_docs: List[Dict] = field(default_factory=list)


@dataclass
class Session:
    """Represents a user session"""
    session_id: str
    created_at: str
    last_active: str
    messages: List[Message] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert session to dictionary"""
        return {
            "session_id": self.session_id,
            "created_at": self.created_at,
            "last_active": self.last_active,
            "messages": [asdict(msg) for msg in self.messages]
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Session':
        """Create session from dictionary"""
        messages = [Message(**msg) for msg in data.get("messages", [])]
        return cls(
            session_id=data["session_id"],
            created_at=data["created_at"],
            last_active=data["last_active"],
            messages=messages
        )


class InMemorySessionStore:
    """In-memory session storage with LRU eviction"""
    
    def __init__(self, max_sessions: int = 1000):
        self.max_sessions = max_sessions
        self.sessions: OrderedDict[str, Session] = OrderedDict()
        logger.info(f"Initialized in-memory session store (max: {max_sessions})")
    
    def create(self, session_id: str) -> Session:
        """Create a new session"""
        now = datetime.now().isoformat()
        session = Session(
            session_id=session_id,
            created_at=now,
            last_active=now,
            messages=[]
        )
        self._store(session_id, session)
        logger.info(f"Created new session: {session_id}")
        return session
    
    def get(self, session_id: str) -> Optional[Session]:
        """Retrieve a session and update its position (LRU)"""
        if session_id in self.sessions:
            # Move to end (most recently used)
            self.sessions.move_to_end(session_id)
            session = self.sessions[session_id]
            logger.debug(f"Retrieved session: {session_id}")
            return session
        logger.debug(f"Session not found: {session_id}")
        return None
    
    def update(self, session: Session) -> None:
        """Update a session"""
        session.last_active = datetime.now().isoformat()
        self._store(session.session_id, session)
        logger.debug(f"Updated session: {session.session_id}")
    
    def delete(self, session_id: str) -> bool:
        """Delete a session"""
        if session_id in self.sessions:
            del self.sessions[session_id]
            logger.info(f"Deleted session from memory: {session_id}")
            return True
        return False
    
    def list_all(self) -> List[Dict[str, Any]]:
        """List all active sessions"""
        return [
            {
                "session_id": session.session_id,
                "created_at": session.created_at,
                "last_active": session.last_active,
                "message_count": len(session.messages)
            }
            for session in self.sessions.values()
        ]
    
    def _store(self, session_id: str, session: Session) -> None:
        """Store session with LRU eviction"""
        # If session exists, remove it first (will be re-added at end)
        if session_id in self.sessions:
            del self.sessions[session_id]
        
        # Check if we need to evict
        if len(self.sessions) >= self.max_sessions:
            # Evict least recently used (first item)
            evicted_id, evicted_session = self.sessions.popitem(last=False)
            logger.warning(f"Session limit reached. Evicted LRU session: {evicted_id}")
        
        # Add session at end (most recently used)
        self.sessions[session_id] = session


class FastAPISessionManager:
    """Manages sessions using FastAPI's native session middleware"""
    
    def __init__(
        self,
        max_sessions: int = 1000,
        session_max_age: int = 1800
    ):
        self.store = InMemorySessionStore(max_sessions)
        self.session_max_age = session_max_age
        logger.info("Session manager initialized (PostgreSQL storage)")
    
    def create_session(self) -> Session:
        """Create a new session with a unique ID"""
        session_id = str(uuid.uuid4())
        session = self.store.create(session_id)
        logger.info(f"Created session {session_id} at {session.created_at}")
        return session
    
    def get_session(self, session_id: str) -> Optional[Session]:
        """Retrieve a session by ID"""
        session = self.store.get(session_id)
        if session:
            # Update last_active timestamp
            session.last_active = datetime.now().isoformat()
            self.store.update(session)
            logger.debug(f"Retrieved session: {session_id}")
        else:
            logger.warning(f"Session not found: {session_id}")
        return session
    
    def update_session(self, session: Session) -> None:
        """Update an existing session"""
        self.store.update(session)
    
    def clear_session(self, session_id: str) -> bool:
        """Clear messages from a session while preserving the session"""
        session = self.store.get(session_id)
        if session:
            session.messages.clear()
            session.last_active = datetime.now().isoformat()
            self.store.update(session)
            logger.info(f"Cleared session: {session_id}")
            return True
        return False
    
    def list_sessions(self) -> List[Dict[str, Any]]:
        """List all active sessions"""
        return self.store.list_all()
    
    def archive_session(self, session: Session) -> str:
        """
        Archive a session (no-op since conversations are saved in real-time)
        
        Note: With the new architecture, conversations are saved to PostgreSQL
        in real-time as they happen, so there's nothing to archive when the
        session expires. This method is kept for compatibility.
        """
        try:
            logger.info(f"Session {session.session_id} expired (conversations already saved in real-time)")
            return f"Session: {session.session_id}"
        except Exception as e:
            logger.error(f"Error in archive_session: {e}", exc_info=True)
            return ""
    
    def cleanup_expired_sessions(self) -> int:
        """Remove sessions that have been inactive for too long"""
        now = datetime.now()
        expired_sessions = []
        
        # Find expired sessions
        for session in list(self.store.sessions.values()):
            last_active = datetime.fromisoformat(session.last_active)
            age = (now - last_active).total_seconds()
            
            if age > self.session_max_age:
                expired_sessions.append(session)
        
        # Archive and delete expired sessions
        for session in expired_sessions:
            logger.info(f"Session expired: {session.session_id} (inactive for {age:.0f}s)")
            self.archive_session(session)
            self.store.delete(session.session_id)
        
        if expired_sessions:
            logger.info(f"Cleaned up {len(expired_sessions)} expired sessions")
        
        return len(expired_sessions)
