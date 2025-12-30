"""PostgreSQL database manager for chatbot sessions"""
import psycopg2
from psycopg2.extras import RealDictCursor
import logging
from typing import Optional, Dict, List, Any
from datetime import datetime
from config import POSTGRES_SCHEMA

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
        self.schema = POSTGRES_SCHEMA
    
    def _get_table_name(self, table: str) -> str:
        """Get fully qualified table name with schema"""
        return f"{self.schema}.{table}"

    def connect(self):
        """Establish database connection"""
        try:
            self.conn = psycopg2.connect(self.connection_string)
            return True
        except Exception as e:
            logger.error(f"Failed to connect to PostgreSQL: {e}")
            return False
    
    def disconnect(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
    
    def authenticate_client(self, company_pin: str, api_key: str) -> Optional[Dict[str, Any]]:
        """
        Authenticate client by validating company_pin and api_key
        
        Args:
            company_pin: Company PIN
            api_key: API key (40-character hash)
            
        Returns:
            Client info dict if authenticated, None otherwise
        """
        try:
            with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    """
                    SELECT client_id, company_pin, api_key, is_active, created_at
                    FROM {} 
                    WHERE company_pin = %s 
                      AND api_key = %s 
                      AND is_active = TRUE
                    """.format(self._get_table_name('clients')),
                    (company_pin, api_key)
                )
                result = cur.fetchone()
                if result:
                    return dict(result)
                else:
                    logger.warning(f"Authentication failed for: {company_pin}")
                    return None
        except Exception as e:
            logger.error(f"Error authenticating client: {e}", exc_info=True)
            return None
    

    def sync_client(self, company_pin: str, api_key: str, is_active: bool = True) -> bool:
        """
        Sync client data from HCMSAPI (INSERT or UPDATE)
        Auto-assigns client_id (1, 2, 3...) in PostgreSQL
        
        Args:
            company_pin: Company PIN (unique identifier)
            api_key: API key (unique identifier)
            is_active: Client active status
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with self.conn.cursor() as cur:
      
                cur.execute(
                    """
                    INSERT INTO {} (company_pin, api_key, is_active, created_at)
                    VALUES (%s, %s, %s, NOW())
                    ON CONFLICT (company_pin) 
                    DO UPDATE SET 
                        api_key = EXCLUDED.api_key,
                        is_active = EXCLUDED.is_active
                    RETURNING client_id
                    """.format(self._get_table_name('clients')),
                    (company_pin, api_key, is_active)
                )
                result = cur.fetchone()
                self.conn.commit()
                return True
        except Exception as e:
            logger.error(f"Error syncing client: {e}", exc_info=True)
            self.conn.rollback()
            return False
    
    def get_active_session_for_user(self, username: str, client_id: int, session_max_age: int) -> Optional[Dict[str, Any]]:
        """
        Get active session for a specific username and client if it exists and hasn't expired
        
        Args:
            username: Username from FlowHCM
            client_id: Client identifier
            session_max_age: Maximum session age in seconds
            
        Returns:
            Session info dict if found and active, None otherwise
        """
        try:
            with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    """
                    SELECT id, session_id, username, fk_client_id, created_at, last_active
                    FROM {} 
                    WHERE username = %s 
                      AND fk_client_id = %s 
                      AND last_active > NOW() - INTERVAL '%s seconds'
                    ORDER BY last_active DESC
                    LIMIT 1
                    """.format(self._get_table_name('sessions')),
                    (username, client_id, session_max_age)
                )
                result = cur.fetchone()
                if result:
                    logger.info(f"Found active session for user {username}: {result['session_id']}")
                    return dict(result)
                else:
                    logger.info(f"No active session found for user {username}")
                    return None
        except Exception as e:
            logger.error(f"Error getting active session for user: {e}", exc_info=True)
            return None

    def create_session(self, session_id: str, username: str, client_id: int) -> Optional[int]:
        """
        Create a new session record
        
        Args:
            session_id: UUID session identifier
            username: Username from FlowHCM
            client_id: Client identifier
            
        Returns:
            Session ID (integer) if successful, None otherwise
        """
        try:
            with self.conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO {} (session_id, username, fk_client_id, created_at, last_active)
                    VALUES (%s, %s, %s, NOW(), NOW())
                    RETURNING id
                    """.format(self._get_table_name('sessions')),
                    (session_id, username, client_id)
                )
                session_db_id = cur.fetchone()[0]
                self.conn.commit()
                logger.info(f"Created new session for user {username}: {session_id}")
                return session_db_id
        except Exception as e:
            logger.error(f"❌ Error creating session in database: {e}", exc_info=True)
            self.conn.rollback()
            return None
    
    def get_session_db_id(self, session_id: str) -> Optional[int]:
        """
        Get the database ID for a session UUID
        
        Args:
            session_id: Session UUID
            
        Returns:
            Database ID (integer) if found, None otherwise
        """
        try:
            with self.conn.cursor() as cur:
                cur.execute(
                    "SELECT id FROM {} WHERE session_id = %s".format(self._get_table_name('sessions')),
                    (session_id,)
                )
                result = cur.fetchone()
                return result[0] if result else None
        except Exception as e:
            logger.error(f"❌ Error getting session DB ID: {e}", exc_info=True)
            return None
    
    def update_session_activity(self, session_id: str) -> bool:
        """Update session last_active timestamp"""
        try:
            logger.debug(f"Updating last_active timestamp for session: {session_id}")
            with self.conn.cursor() as cur:
                cur.execute(
                    "UPDATE {} SET last_active = NOW() WHERE session_id = %s".format(self._get_table_name('sessions')),
                    (session_id,)
                )
                self.conn.commit()
                logger.info(f"✅ Session activity updated in database (session: {session_id})")
                return True
        except Exception as e:
            logger.error(f"❌ Error updating session activity in database: {e}", exc_info=True)
            self.conn.rollback()
            return False
    
    def _get_current_conversation_table(self) -> str:
        """
        Get the current month's conversation table name
        
        Returns:
            Table name like 'conversation_dec_2025'
        """
        month_abbr = datetime.now().strftime("%b").lower()  
        year = datetime.now().strftime("%Y")  
        return f"conversation_{month_abbr}_{year}"
    
    def _ensure_conversation_table_exists(self, table_name: str) -> bool:
        """
        Create conversation table for the month if it doesn't exist
        
        Args:
            table_name: Table name like 'conversation_december'
            
        Returns:
            True if table exists or was created, False on error
        """
        try:
            with self.conn.cursor() as cur:
                cur.execute(
                    f"""
                    CREATE TABLE IF NOT EXISTS {self._get_table_name(table_name)} (
                        conversation_id SERIAL PRIMARY KEY,
                        fk_session_id INTEGER NOT NULL,
                        user_message TEXT NOT NULL,
                        chatbot_response TEXT NOT NULL,
                        tokens_used INTEGER DEFAULT 0,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (fk_session_id) REFERENCES {self._get_table_name('sessions')}(id) ON DELETE CASCADE
                    )
                    """
                )
                # Create indexes
                cur.execute(
                    f"""
                    CREATE INDEX IF NOT EXISTS idx_{table_name}_session_id 
                    ON {self._get_table_name(table_name)}(fk_session_id)
                    """
                )
                cur.execute(
                    f"""
                    CREATE INDEX IF NOT EXISTS idx_{table_name}_created_at 
                    ON {self._get_table_name(table_name)}(created_at)
                    """
                )
                self.conn.commit()
                logger.info(f"✅ Ensured conversation table exists: {table_name}")
                return True
        except Exception as e:
            logger.error(f"❌ Error creating conversation table {table_name}: {e}", exc_info=True)
            self.conn.rollback()
            return False
    
    def add_conversation(self, session_db_id: int, user_message: str, chatbot_response: str, tokens_used: int = 0) -> bool:
        """
        Add a conversation entry (user message + chatbot response) to current month's table
        
        Args:
            session_db_id: Session database ID (integer from sessions.id)
            user_message: User's message
            chatbot_response: Chatbot's response
            
        Returns:
            True if successful, False otherwise
        """
        try:
            table_name = self._get_current_conversation_table()
            
            if not self._ensure_conversation_table_exists(table_name):
                return False
            
            logger.debug(f"Inserting conversation into {table_name} for session_id: {session_db_id}")
            logger.debug(f"  User message: {user_message[:100]}...")
            logger.debug(f"  Bot response: {chatbot_response[:100]}...")
            
            with self.conn.cursor() as cur:
                cur.execute(
                    f"""
                    INSERT INTO {self._get_table_name(table_name)} (fk_session_id, user_message, chatbot_response, tokens_used, created_at)
                    VALUES (%s, %s, %s, %s, NOW())
                    """,
                    (session_db_id, user_message, chatbot_response, tokens_used)
                )
                self.conn.commit()
                logger.info(f"✅ Conversation added to {table_name} (session_id: {session_db_id}, tokens: {tokens_used})")
                return True
        except Exception as e:
            logger.error(f"❌ Error adding conversation to database: {e}", exc_info=True)
            self.conn.rollback()
            return False
    

    
    # Token Management Methods
    
    def get_client_token_limit(self, client_id: int) -> Optional[int]:
        """
        Get token limit per month for a client
        
        Args:
            client_id: Client ID
            
        Returns:
            Token limit or None if not found
        """
        try:
            with self.conn.cursor() as cur:
                cur.execute(
                    "SELECT token_limit_per_month FROM {} WHERE client_id = %s".format(self._get_table_name('clients')),
                    (client_id,)
                )
                result = cur.fetchone()
                return result[0] if result else None
        except Exception as e:
            logger.error(f"Error getting client token limit: {e}", exc_info=True)
            return None
    
    def get_client_token_usage(self, client_id: int, month_year: str = None) -> int:
        """
        Get current token usage for a client in a specific month
        
        Args:
            client_id: Client ID
            month_year: Month and year (e.g., 'dec_2025'), defaults to current month
            
        Returns:
            Total tokens used in the month
        """
        try:
            if month_year is None:
                month_abbr = datetime.now().strftime("%b").lower()
                year = datetime.now().strftime("%Y")
                month_year = f"{month_abbr}_{year}"
            
            with self.conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT tokens_used FROM {} 
                    WHERE fk_client_id = %s AND month_year = %s
                    """.format(self._get_table_name('tokens')),
                    (client_id, month_year)
                )
                result = cur.fetchone()
                return result[0] if result else 0
        except Exception as e:
            logger.error(f"Error getting client token usage: {e}", exc_info=True)
            return 0
    
    def update_token_usage(self, client_id: int, tokens_used: int, month_year: str = None) -> bool:
        """
        Update token usage for a client (add to existing usage)
        
        Args:
            client_id: Client ID
            tokens_used: Number of tokens to add
            month_year: Month and year (e.g., 'dec_2025'), defaults to current month
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if month_year is None:
                month_abbr = datetime.now().strftime("%b").lower()
                year = datetime.now().strftime("%Y")
                month_year = f"{month_abbr}_{year}"
            
            logger.debug(f"Updating token usage for client {client_id}: +{tokens_used} tokens for {month_year}")
            
            with self.conn.cursor() as cur:
                # Insert or update token usage
                cur.execute(
                    """
                    INSERT INTO {} (fk_client_id, month_year, tokens_used, created_at, updated_at)
                    VALUES (%s, %s, %s, NOW(), NOW())
                    ON CONFLICT (fk_client_id, month_year)
                    DO UPDATE SET 
                        tokens_used = {}.tokens_used + EXCLUDED.tokens_used,
                        updated_at = NOW()
                    """.format(self._get_table_name('tokens'), self._get_table_name('tokens')),
                    (client_id, month_year, tokens_used)
                )
                self.conn.commit()
                logger.info(f"✅ Token usage updated for client {client_id}: +{tokens_used} tokens ({month_year})")
                return True
        except Exception as e:
            logger.error(f"❌ Error updating token usage: {e}", exc_info=True)
            self.conn.rollback()
            return False
    
    def check_token_limit(self, client_id: int) -> Dict[str, Any]:
        """
        Check if client has exceeded token limit for current month
        
        Args:
            client_id: Client ID
            
        Returns:
            Dict with 'allowed' (bool), 'usage' (int), 'limit' (int), 'remaining' (int)
        """
        try:
            limit = self.get_client_token_limit(client_id)
            usage = self.get_client_token_usage(client_id)
            
            if limit is None:
                limit = 100000  # Default limit
            
            remaining = max(0, limit - usage)
            allowed = usage < limit
            
            return {
                'allowed': allowed,
                'usage': usage,
                'limit': limit,
                'remaining': remaining
            }
        except Exception as e:
            logger.error(f"Error checking token limit: {e}", exc_info=True)
            return {
                'allowed': True,  # Allow on error to avoid blocking
                'usage': 0,
                'limit': 100000,
                'remaining': 100000
            }
    
    # Document Management Methods
    
    def add_document(self, announcement_id: int, client_id: int, file_name: str, 
                     document_title: str, file_extension: str, chunk_count: int) -> Optional[int]:
        """
        Add or update document metadata
        
        Args:
            announcement_id: Announcement ID from HCMS
            client_id: Client ID
            file_name: Original filename
            document_title: Document title
            file_extension: File extension (pdf, docx, etc)
            chunk_count: Number of chunks created
            
        Returns:
            Document ID if successful, None otherwise
        """
        try:
            logger.info("="*60)
            logger.info("💾 SAVING DOCUMENT METADATA TO DATABASE")
            logger.info(f"   Announcement ID: {announcement_id}")
            logger.info(f"   Client ID: {client_id}")
            logger.info(f"   File Name: {file_name}")
            logger.info(f"   Document Title: {document_title}")
            logger.info(f"   File Extension: {file_extension}")
            logger.info(f"   Chunk Count: {chunk_count}")
            
            with self.conn.cursor() as cur:
                cur.execute(
                    "SELECT document_id FROM {} WHERE announcement_id = %s AND fk_client_id = %s".format(self._get_table_name('documents')),
                    (announcement_id, client_id)
                )
                existing = cur.fetchone()
                
                cur.execute(
                    """
                    INSERT INTO {} 
                    (announcement_id, fk_client_id, file_name, document_title, file_extension, chunk_count, synced_at)
                    VALUES (%s, %s, %s, %s, %s, %s, NOW())
                    ON CONFLICT (announcement_id, fk_client_id)
                    DO UPDATE SET
                        file_name = EXCLUDED.file_name,
                        document_title = EXCLUDED.document_title,
                        file_extension = EXCLUDED.file_extension,
                        chunk_count = EXCLUDED.chunk_count,
                        synced_at = NOW()
                    RETURNING document_id
                    """.format(self._get_table_name('documents')),
                    (announcement_id, client_id, file_name, document_title, file_extension, chunk_count)
                )
                document_id = cur.fetchone()[0]
                self.conn.commit()
                
                logger.info("✅ Document metadata saved successfully")
                logger.info(f"   Document ID: {document_id}")
                logger.info(f"   Table: {self._get_table_name('documents')}")
                logger.info(f"   Action: {'Updated' if existing else 'Inserted'}")
                logger.info("="*60)
                
                return document_id
        except Exception as e:
            logger.error("❌ Error adding document to database")
            logger.error(f"   Error type: {type(e).__name__}")
            logger.error(f"   Error details: {str(e)}", exc_info=True)
            self.conn.rollback()
            return None
    

    
    def get_user_session_history(self, username: str, client_id: int, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get session history for a specific user
        
        Args:
            username: Username from FlowHCM
            client_id: Client identifier
            limit: Maximum number of sessions to return
            
        Returns:
            List of session info dictionaries
        """
        try:
            with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    """
                    SELECT id, session_id, username, fk_client_id, created_at, last_active
                    FROM {} 
                    WHERE username = %s AND fk_client_id = %s
                    ORDER BY last_active DESC
                    LIMIT %s
                    """.format(self._get_table_name('sessions')),
                    (username, client_id, limit)
                )
                results = cur.fetchall()
                return [dict(row) for row in results]
        except Exception as e:
            logger.error(f"Error getting user session history: {e}", exc_info=True)
            return []

    def get_session_analytics(self, client_id: int = None, days: int = 30) -> Dict[str, Any]:
        """
        Get session analytics for audit and monitoring purposes
        
        Args:
            client_id: Optional client ID to filter by specific client
            days: Number of days to look back (default: 30)
            
        Returns:
            Dictionary with session analytics
        """
        try:
            with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
                # Base query conditions
                where_clause = "WHERE created_at >= NOW() - INTERVAL '%s days'"
                params = [days]
                
                if client_id:
                    where_clause += " AND fk_client_id = %s"
                    params.append(client_id)
                
                # Total sessions
                cur.execute(f"SELECT COUNT(*) as total_sessions FROM {self._get_table_name('sessions')} {where_clause}", params)
                total_sessions = cur.fetchone()['total_sessions']
                
                # Active sessions (not expired)
                active_where = where_clause + " AND last_active > NOW() - INTERVAL '%s seconds'"
                active_params = params + [1800]  # SESSION_MAX_AGE
                cur.execute(f"SELECT COUNT(*) as active_sessions FROM {self._get_table_name('sessions')} {active_where}", active_params)
                active_sessions = cur.fetchone()['active_sessions']
                
                # Unique users
                cur.execute(f"SELECT COUNT(DISTINCT username) as unique_users FROM {self._get_table_name('sessions')} {where_clause}", params)
                unique_users = cur.fetchone()['unique_users']
                
                # Sessions by day
                cur.execute(f"""
                    SELECT DATE(created_at) as session_date, COUNT(*) as session_count
                    FROM {self._get_table_name('sessions')} {where_clause}
                    GROUP BY DATE(created_at)
                    ORDER BY session_date DESC
                    LIMIT 7
                """, params)
                sessions_by_day = [dict(row) for row in cur.fetchall()]
                
                return {
                    'total_sessions': total_sessions,
                    'active_sessions': active_sessions,
                    'expired_sessions': total_sessions - active_sessions,
                    'unique_users': unique_users,
                    'sessions_by_day': sessions_by_day,
                    'period_days': days,
                    'client_id': client_id
                }
                
        except Exception as e:
            logger.error(f"Error getting session analytics: {e}", exc_info=True)
            return {
                'total_sessions': 0,
                'active_sessions': 0,
                'expired_sessions': 0,
                'unique_users': 0,
                'sessions_by_day': [],
                'period_days': days,
                'client_id': client_id,
                'error': str(e)
            }

    def delete_client_documents(self, client_id: int) -> bool:
        """
        Delete all document records for a specific client
        Used when rebuilding index from scratch
        
        Args:
            client_id: Client ID
            
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info(f"🗑️  Deleting all document records for client_id: {client_id}")
            with self.conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM {} WHERE fk_client_id = %s".format(self._get_table_name('documents')),
                    (client_id,)
                )
                deleted_count = cur.rowcount
                self.conn.commit()
                logger.info(f"✅ Deleted {deleted_count} document record(s)")
                return True
        except Exception as e:
            logger.error(f"❌ Error deleting client documents: {e}", exc_info=True)
            self.conn.rollback()
            return False
