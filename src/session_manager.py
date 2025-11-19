"""Redis-based session management for multi-user support"""
import redis
import json
import logging
import os
from typing import Optional, Dict, List, Any
from datetime import datetime
from config import REDIS_HOST, REDIS_PORT, REDIS_DB, SESSION_EXPIRY_SECONDS, ARCHIVE_FOLDER, ARCHIVE_FORMAT

logger = logging.getLogger(__name__)


class SessionManager:
    """Manages user sessions with Redis"""
    
    def __init__(
        self, 
        host: str = REDIS_HOST,
        port: int = REDIS_PORT,
        db: int = REDIS_DB,
        expiry: int = SESSION_EXPIRY_SECONDS
    ):
        """Initialize Redis connection"""
        self.expiry = expiry
        self.archive_folder = ARCHIVE_FOLDER
        
        # Create archive folder if it doesn't exist
        os.makedirs(self.archive_folder, exist_ok=True)
        
        try:
            self.redis_client = redis.Redis(
                host=host,
                port=port,
                db=db,
                decode_responses=True,
                socket_connect_timeout=5
            )
            # Test connection
            self.redis_client.ping()
            logger.info(f"✅ Connected to Redis at {host}:{port}")
            logger.info(f"📁 Session archives will be saved to: {self.archive_folder}")
        except redis.ConnectionError as e:
            logger.error(f"❌ Failed to connect to Redis: {e}")
            raise
    
    def _get_key(self, session_id: str) -> str:
        """Generate Redis key for session"""
        return f"session:{session_id}"
    
    def create_session(self, session_id: str) -> Dict[str, Any]:
        """Create a new session"""
        session_data = {
            "session_id": session_id,
            "messages": [],
            "created_at": datetime.now().isoformat(),
            "last_active": datetime.now().isoformat()
        }
        
        key = self._get_key(session_id)
        self.redis_client.setex(
            key,
            self.expiry,
            json.dumps(session_data)
        )
        
        logger.info(f"Created new session: {session_id}")
        return session_data
    
    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve session data"""
        key = self._get_key(session_id)
        data = self.redis_client.get(key)
        
        if data:
            session_data = json.loads(data)
            # Refresh expiry on access
            self.redis_client.expire(key, self.expiry)
            logger.debug(f"Retrieved session: {session_id}")
            return session_data
        
        logger.debug(f"Session not found: {session_id}")
        return None
    
    def update_session(self, session_id: str, session_data: Dict[str, Any]) -> bool:
        """Update session data"""
        session_data["last_active"] = datetime.now().isoformat()
        
        key = self._get_key(session_id)
        self.redis_client.setex(
            key,
            self.expiry,
            json.dumps(session_data)
        )
        
        logger.debug(f"Updated session: {session_id}")
        return True
    
    def archive_session(self, session_id: str) -> bool:
        """Archive session to file before deletion"""
        session_data = self.get_session(session_id)
        
        if not session_data:
            logger.warning(f"Cannot archive - session not found: {session_id}")
            return False
        
        try:
            # Add end timestamp
            session_data["ended_at"] = datetime.now().isoformat()
            
            # Create filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"session_{session_id[:8]}_{timestamp}.json"
            filepath = os.path.join(self.archive_folder, filename)
            
            # Save to file
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(session_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"📁 Archived session to: {filename}")
            return True
        
        except Exception as e:
            logger.error(f"Failed to archive session {session_id}: {e}")
            return False
    
    def delete_session(self, session_id: str, archive: bool = True) -> bool:
        """Delete a session (archives by default before deletion)"""
        # Archive first if requested
        if archive:
            self.archive_session(session_id)
        
        # Delete from Redis
        key = self._get_key(session_id)
        result = self.redis_client.delete(key)
        
        if result:
            logger.info(f"🗑️ Deleted session from Redis: {session_id}")
            return True
        return False
    
    def session_exists(self, session_id: str) -> bool:
        """Check if session exists"""
        key = self._get_key(session_id)
        return self.redis_client.exists(key) > 0
    
    def add_message(
        self, 
        session_id: str, 
        role: str, 
        content: str,
        context_docs: Optional[List[Dict]] = None
    ) -> bool:
        """Add a message to session history"""
        session_data = self.get_session(session_id)
        
        if not session_data:
            session_data = self.create_session(session_id)
        
        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
            "context_docs": context_docs or []
        }
        
        session_data["messages"].append(message)
        return self.update_session(session_id, session_data)
    
    def get_messages(self, session_id: str) -> List[Dict[str, Any]]:
        """Get all messages for a session"""
        session_data = self.get_session(session_id)
        
        if session_data:
            return session_data.get("messages", [])
        return []
    
    def get_recent_context(
        self, 
        session_id: str, 
        num_exchanges: int = 2
    ) -> str:
        """Get recent conversation context"""
        messages = self.get_messages(session_id)
        recent_messages = messages[-(num_exchanges * 2):]
        
        context = ""
        for msg in recent_messages:
            role = "Human" if msg["role"] == "user" else "Assistant"
            context += f"{role}: {msg['content']}\n"
        
        return context
    
    def clear_session(self, session_id: str) -> bool:
        """Clear session messages but keep session alive"""
        session_data = self.get_session(session_id)
        
        if session_data:
            session_data["messages"] = []
            return self.update_session(session_id, session_data)
        
        return False
    
    def get_active_sessions(self) -> List[str]:
        """Get all active session IDs"""
        pattern = "session:*"
        keys = self.redis_client.keys(pattern)
        return [key.replace("session:", "") for key in keys]
    
    def get_session_count(self) -> int:
        """Get count of active sessions"""
        return len(self.get_active_sessions())
    
    def end_session(self, session_id: str) -> bool:
        """End a session - archives and deletes from Redis"""
        logger.info(f"Ending session: {session_id}")
        return self.delete_session(session_id, archive=True)
