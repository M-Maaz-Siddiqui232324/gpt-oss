"""PostgreSQL database manager for chatbot sessions"""
import psycopg2
from psycopg2.extras import RealDictCursor
import logging
from typing import Optional, Dict, List, Any
from datetime import datetime

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
    
    def authenticate_client(self, company_pin: str, api_key: str) -> Optional[Dict[str, Any]]:
        """
        Authenticate client by validating company_pin and api_key
        
        Args:
            company_pin: Company PIN
            api_key: API key (with FLOW-{client_id}- prefix)
            
        Returns:
            Client info dict if authenticated, None otherwise
        """
        try:
            # Extract the actual key part (remove FLOW-{client_id}- prefix)
            # Format: FLOW-1-test123456789... -> test123456789...
            if api_key.startswith("FLOW-"):
                # Split by '-' and take everything after the second dash
                parts = api_key.split("-", 2)  # Split into max 3 parts: ['FLOW', '1', 'test123...']
                if len(parts) >= 3:
                    actual_key = parts[2]  # Get the part after FLOW-{client_id}-
                    logger.info(f"🔍 Extracted API key: '{api_key}' -> '{actual_key[:30]}...'")
                else:
                    actual_key = api_key
                    logger.warning(f"⚠️  API key format unexpected: '{api_key}'")
            else:
                actual_key = api_key
                logger.warning(f"⚠️  API key doesn't start with FLOW-: '{api_key[:30]}...'")
            
            logger.info(f"🔍 Authenticating with:")
            logger.info(f"   Company PIN: '{company_pin}' (length: {len(company_pin)})")
            logger.info(f"   API Key (extracted): '{actual_key[:30]}...' (length: {len(actual_key)})")
            
            with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
                # Authenticate using extracted key
                cur.execute(
                    """
                    SELECT client_id, company_pin, api_key, is_active, created_at
                    FROM chatbot.clients 
                    WHERE company_pin = %s 
                      AND api_key = %s 
                      AND is_active = TRUE
                    """,
                    (company_pin, actual_key)
                )
                result = cur.fetchone()
                if result:
                    logger.info(f"✅ Client authenticated successfully from database (client_id: {result['client_id']})")
                    return dict(result)
                else:
                    logger.warning(f"❌ Authentication failed - no matching company_pin and api_key combination in database")
                    logger.warning(f"   Searched for: pin='{company_pin}', key='{actual_key[:30]}...'")
                    return None
        except Exception as e:
            logger.error(f"❌ Error authenticating client: {e}", exc_info=True)
            return None
    
    def get_client_by_api_key(self, api_key: str) -> Optional[Dict[str, Any]]:
        """Get client by API key (for backward compatibility)"""
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
    
    def sync_client(self, client_id: int, company_pin: str, api_key: str, is_active: bool = True) -> bool:
        """
        Sync client data from HCMSAPI (INSERT or UPDATE)
        
        Args:
            client_id: Client identifier
            company_pin: Company PIN
            api_key: API key
            is_active: Client active status
            
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info(f"Syncing client to database: client_id={client_id}, company_pin={company_pin}, api_key={api_key[:20]}..., is_active={is_active}")
            with self.conn.cursor() as cur:
                # UPSERT: Insert or update if exists
                cur.execute(
                    """
                    INSERT INTO chatbot.clients (client_id, company_pin, api_key, is_active, created_at)
                    VALUES (%s, %s, %s, %s, NOW())
                    ON CONFLICT (client_id) 
                    DO UPDATE SET 
                        company_pin = EXCLUDED.company_pin,
                        api_key = EXCLUDED.api_key,
                        is_active = EXCLUDED.is_active
                    """,
                    (client_id, company_pin, api_key, is_active)
                )
                self.conn.commit()
                logger.info(f"✅ Client {client_id} synced to PostgreSQL database (table: chatbot.clients)")
                return True
        except Exception as e:
            logger.error(f"❌ Error syncing client to database: {e}", exc_info=True)
            self.conn.rollback()
            return False
    
    def create_session(self, session_id: str, username: str, client_id: int) -> bool:
        """
        Create a new session record
        
        Args:
            session_id: UUID session identifier
            username: Username from FlowHCM
            client_id: Client identifier
            
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.debug(f"Inserting session into database: session_id={session_id}, username={username}, client_id={client_id}")
            with self.conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO chatbot.sessions (session_id, username, fk_client_id, created_at, last_active)
                    VALUES (%s, %s, %s, NOW(), NOW())
                    """,
                    (session_id, username, client_id)
                )
                self.conn.commit()
                logger.info(f"✅ Session created in database: {session_id} (user={username}, client={client_id})")
                return True
        except Exception as e:
            logger.error(f"❌ Error creating session in database: {e}", exc_info=True)
            self.conn.rollback()
            return False
    
    def update_session_activity(self, session_id: str) -> bool:
        """Update session last_active timestamp"""
        try:
            logger.debug(f"Updating last_active timestamp for session: {session_id}")
            with self.conn.cursor() as cur:
                cur.execute(
                    "UPDATE chatbot.sessions SET last_active = NOW() WHERE session_id = %s",
                    (session_id,)
                )
                self.conn.commit()
                logger.info(f"✅ Session activity updated in database (session: {session_id})")
                return True
        except Exception as e:
            logger.error(f"❌ Error updating session activity in database: {e}", exc_info=True)
            self.conn.rollback()
            return False
    
    def add_conversation(self, session_id: str, user_message: str, chatbot_response: str) -> bool:
        """
        Add a conversation entry (user message + chatbot response)
        
        Args:
            session_id: Session UUID
            user_message: User's message
            chatbot_response: Chatbot's response
            
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.debug(f"Inserting conversation into database for session: {session_id}")
            logger.debug(f"  User message: {user_message[:100]}...")
            logger.debug(f"  Bot response: {chatbot_response[:100]}...")
            with self.conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO chatbot.conversation (fk_session_id, user_message, chatbot_response, created_at)
                    VALUES (%s, %s, %s, NOW())
                    """,
                    (session_id, user_message, chatbot_response)
                )
                self.conn.commit()
                logger.info(f"✅ Conversation added to database (session: {session_id})")
                return True
        except Exception as e:
            logger.error(f"❌ Error adding conversation to database: {e}", exc_info=True)
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
    
    def get_session_conversations(self, session_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Get conversation history for a session
        
        Args:
            session_id: Session UUID
            limit: Maximum number of conversations to retrieve
            
        Returns:
            List of conversation dictionaries
        """
        try:
            with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    """
                    SELECT conversation_id, fk_session_id, user_message, chatbot_response, created_at
                    FROM chatbot.conversation 
                    WHERE fk_session_id = %s 
                    ORDER BY created_at ASC
                    LIMIT %s
                    """,
                    (session_id, limit)
                )
                results = cur.fetchall()
                return [dict(row) for row in results]
        except Exception as e:
            logger.error(f"Error getting session conversations: {e}", exc_info=True)
            return []
    
    def get_client_sessions(self, client_id: int, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent sessions for a client"""
        try:
            with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    """
                    SELECT s.session_id, s.username, s.fk_client_id, s.created_at, s.last_active,
                           COUNT(c.conversation_id) as message_count
                    FROM chatbot.sessions s
                    LEFT JOIN chatbot.conversation c ON c.fk_session_id = s.session_id
                    WHERE s.fk_client_id = %s 
                    GROUP BY s.session_id, s.username, s.fk_client_id, s.created_at, s.last_active
                    ORDER BY s.last_active DESC 
                    LIMIT %s
                    """,
                    (client_id, limit)
                )
                results = cur.fetchall()
                return [dict(row) for row in results]
        except Exception as e:
            logger.error(f"Error getting client sessions: {e}", exc_info=True)
            return []
