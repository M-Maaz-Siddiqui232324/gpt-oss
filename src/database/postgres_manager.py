"""PostgreSQL database manager for chatbot sessions"""
import psycopg2
from psycopg2.extras import RealDictCursor, Json
import logging
from typing import Optional, Dict, List, Any
from datetime import datetime
import json

logger = logging.getLogger(__name__)


class PostgresManager:
    """Manages PostgreSQL database connections and operations"""
    
    def __init__(self, connection_string: str):
        """
        Initialize PostgreSQL manager
        
        Args:
            connection_string: PostgreSQL connection string
                Format: "host=localhost port=5432 dbname=chatbot_db user=postgres password=yourpassword"
        """
        self.connection_string = connection_string
        self.conn = None
        logger.info("PostgreSQL manager initialized")
    
    def connect(self):
        """Establish database connection"""
        try:
            self.conn = psycopg2.connect(self.connection_string)
            logger.info("Connected to PostgreSQL database")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to PostgreSQL: {e}", exc_info=True)
            return False
    
    def disconnect(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
            logger.info("Disconnected from PostgreSQL database")
    
    def get_client_by_api_key(self, api_key: str) -> Optional[Dict[str, Any]]:
        """Get client by API key"""
        try:
            with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    "SELECT * FROM chatbot.clients WHERE api_key = %s AND is_active = TRUE",
                    (api_key,)
                )
                result = cur.fetchone()
                return dict(result) if result else None
        except Exception as e:
            logger.error(f"Error getting client by API key: {e}", exc_info=True)
            return None
    
    def create_session(self, session_id: str, client_id: int, session_start_time: str) -> bool:
        """Create a new session record"""
        try:
            with self.conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO chatbot.sessions (session_id, client_id, messages, session_start_time)
                    VALUES (%s, %s, %s, %s)
                    """,
                    (session_id, client_id, Json([]), session_start_time)
                )
                self.conn.commit()
                logger.info(f"Created session record: {session_id}")
                return True
        except Exception as e:
            logger.error(f"Error creating session: {e}", exc_info=True)
            self.conn.rollback()
            return False
    
    def update_session_messages(self, session_id: str, messages: List[Dict[str, Any]]) -> bool:
        """Update session messages"""
        try:
            with self.conn.cursor() as cur:
                cur.execute(
                    "UPDATE chatbot.sessions SET messages = %s WHERE session_id = %s",
                    (Json(messages), session_id)
                )
                self.conn.commit()
                logger.debug(f"Updated messages for session: {session_id}")
                return True
        except Exception as e:
            logger.error(f"Error updating session messages: {e}", exc_info=True)
            self.conn.rollback()
            return False
    
    def end_session(self, session_id: str, client_id: int, messages: List[Dict[str, Any]], 
                    session_start_time: str, session_end_time: str) -> bool:
        """End a session and store final messages (INSERT or UPDATE)"""
        try:
            with self.conn.cursor() as cur:
                # Use UPSERT (INSERT ... ON CONFLICT UPDATE)
                cur.execute(
                    """
                    INSERT INTO chatbot.sessions (session_id, client_id, messages, session_start_time, session_end_time)
                    VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT (session_id) 
                    DO UPDATE SET 
                        messages = EXCLUDED.messages,
                        session_end_time = EXCLUDED.session_end_time
                    """,
                    (session_id, client_id, Json(messages), session_start_time, session_end_time)
                )
                self.conn.commit()
                logger.info(f"Ended session: {session_id}")
                return True
        except Exception as e:
            logger.error(f"Error ending session: {e}", exc_info=True)
            self.conn.rollback()
            return False
    
    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session by ID"""
        try:
            with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    "SELECT * FROM chatbot.sessions WHERE session_id = %s",
                    (session_id,)
                )
                result = cur.fetchone()
                return dict(result) if result else None
        except Exception as e:
            logger.error(f"Error getting session: {e}", exc_info=True)
            return None
    
    def get_client_sessions(self, client_id: int, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent sessions for a client"""
        try:
            with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    """
                    SELECT session_id, session_start_time, session_end_time,
                           jsonb_array_length(messages) as message_count
                    FROM chatbot.sessions 
                    WHERE client_id = %s 
                    ORDER BY session_start_time DESC 
                    LIMIT %s
                    """,
                    (client_id, limit)
                )
                results = cur.fetchall()
                return [dict(row) for row in results]
        except Exception as e:
            logger.error(f"Error getting client sessions: {e}", exc_info=True)
            return []
    

