"""FastAPI server for the RAG chatbot"""
import logging
import os
import sys
from datetime import datetime
from fastapi import FastAPI, HTTPException, Request, Header
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.sessions import SessionMiddleware
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
import secrets
import uuid
from contextvars import ContextVar

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from config import *
from rag_system import RAGSystem
from fastapi_session_manager import FastAPISessionManager
import utils

# Context variable to track current client
current_client_context: ContextVar[Optional[str]] = ContextVar('current_client_context', default=None)

# Setup logging directories
logs_dir = os.path.join(os.path.dirname(__file__), '..', 'logs')
os.makedirs(logs_dir, exist_ok=True)


class ClientContextFilter(logging.Filter):
    """Filter that routes logs to client-specific files based on context"""
    
    def filter(self, record):
        # Add client context to record
        record.client_pin = current_client_context.get()
        return True


class ClientAwareHandler(logging.Handler):
    """Handler that routes logs to client-specific folders"""
    
    def __init__(self):
        super().__init__()
        self.client_handlers = {}
        self.formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    def emit(self, record):
        try:
            client_pin = getattr(record, 'client_pin', None)
            
            if client_pin:
                # Route to specific client log
                handler = self._get_client_handler(client_pin)
                handler.emit(record)
            else:
                # System event - write to ALL existing client folders
                self._emit_to_all_clients(record)
        except Exception:
            self.handleError(record)
    
    def _get_client_handler(self, client_pin: str):
        """Get or create handler for client"""
        today = datetime.now().strftime('%Y-%m-%d')
        cache_key = f"{client_pin}_{today}"
        
        if cache_key not in self.client_handlers:
            # Create client folder
            client_dir = os.path.join(logs_dir, client_pin)
            os.makedirs(client_dir, exist_ok=True)
            
            # Create file handler
            log_file = os.path.join(client_dir, f"{today}.log")
            handler = logging.FileHandler(log_file, encoding='utf-8')
            handler.setFormatter(self.formatter)
            self.client_handlers[cache_key] = handler
        
        return self.client_handlers[cache_key]
    
    def _emit_to_all_clients(self, record):
        """Write system events to all existing client folders"""
        # Get all client folders
        if not os.path.exists(logs_dir):
            return
        
        for item in os.listdir(logs_dir):
            item_path = os.path.join(logs_dir, item)
            if os.path.isdir(item_path):
                # This is a client folder
                handler = self._get_client_handler(item)
                handler.emit(record)


# Setup root logger
root_logger = logging.getLogger()
root_logger.setLevel(getattr(logging, LOG_LEVEL))
root_logger.handlers = []

# Add client-aware handler
client_aware_handler = ClientAwareHandler()
client_aware_handler.addFilter(ClientContextFilter())
root_logger.addHandler(client_aware_handler)

# Add console handler
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
root_logger.addHandler(console_handler)

logger = logging.getLogger(__name__)


def set_client_context(company_pin: str):
    """Set the current client context for logging"""
    current_client_context.set(company_pin)


def clear_client_context():
    """Clear the current client context"""
    current_client_context.set(None)

# Validate SECRET_KEY
if not SECRET_KEY or len(SECRET_KEY) < 32:
    logger.warning("SECRET_KEY not set or too short. Generating a random key for development.")
    SECRET_KEY = secrets.token_urlsafe(32)
    logger.warning(f"Generated SECRET_KEY: {SECRET_KEY}")
    logger.warning("Set this in your environment for production!")

# Initialize FastAPI
app = FastAPI(
    title="FlowHCM RAG Chatbot API",
    description="Local RAG chatbot with GPT-OSS-20B",
    version="1.0.0"
)

# Session Middleware (must be added before CORS)
app.add_middleware(
    SessionMiddleware,
    secret_key=SECRET_KEY,
    session_cookie="session",
    max_age=SESSION_MAX_AGE,
    same_site="lax",
    https_only=False  # Set to True in production with HTTPS
)

# CORS - Must specify exact origins when using credentials
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global RAG system and session manager
rag_system: Optional[RAGSystem] = None
session_manager: Optional[FastAPISessionManager] = None


# Request/Response models
class QueryRequest(BaseModel):
    query: str
    max_tokens: Optional[int] = DEFAULT_MAX_TOKENS
    temperature: Optional[float] = DEFAULT_TEMPERATURE
    top_p: Optional[float] = DEFAULT_TOP_P


class SourceDocument(BaseModel):
    content: str
    source_file: str
    chunk_id: int
    relevance_score: float


class QueryResponse(BaseModel):
    response: str
    session_id: str


class SystemStatus(BaseModel):
    status: str
    documents_loaded: int
    chunks_created: int
    model_loaded: bool


@app.on_event("startup")
async def startup_event():
    """Initialize RAG system and session manager on startup"""
    global rag_system, session_manager
    logger.info("="*60)
    logger.info("Starting FastAPI server")
    logger.info("="*60)
    
    try:
        # Initialize session manager
        session_manager = FastAPISessionManager(
            max_sessions=MAX_SESSIONS,
            session_max_age=SESSION_MAX_AGE
        )
        logger.info("Session manager initialized successfully")
        
        # Initialize RAG system (without loading any client-specific index)
        rag_system = RAGSystem()
        logger.info("RAG system initialized (per-client mode)")
        logger.info("Note: Client-specific indexes will be loaded on-demand")
        
        # Start background cleanup task
        import asyncio
        asyncio.create_task(cleanup_sessions_task())
        logger.info(f"Started session cleanup task (interval: {CLEANUP_INTERVAL}s)")
    
    except Exception as e:
        logger.error(f"Startup error: {e}", exc_info=True)
        raise


async def cleanup_sessions_task():
    """Background task to clean up expired sessions"""
    import asyncio
    while True:
        await asyncio.sleep(CLEANUP_INTERVAL)
        try:
            if session_manager:
                count = session_manager.cleanup_expired_sessions()
                if count > 0:
                    logger.info(f"Cleanup task removed {count} expired sessions")
        except Exception as e:
            logger.error(f"Error in cleanup task: {e}", exc_info=True)


@app.get("/", response_model=dict)
async def root():
    """Root endpoint"""
    return {
        "message": "FlowHCM RAG Chatbot API",
        "version": "1.0.0",
        "status": "running"
    }


@app.get("/health", response_model=SystemStatus)
async def health_check():
    """Health check endpoint"""
    if rag_system is None:
        raise HTTPException(status_code=503, detail="RAG system not initialized")
    
    return SystemStatus(
        status="healthy",
        documents_loaded=len(rag_system.documents),
        chunks_created=len(rag_system.chunks),
        model_loaded=rag_system.llm_engine.model is not None
    )


@app.post("/query", response_model=QueryResponse)
async def query(
    request_body: QueryRequest, 
    request: Request,
    x_client_id: Optional[str] = Header(None, alias="X-Client-ID"),
    x_company_pin: Optional[str] = Header(None, alias="X-Company-Pin"),
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
    x_user_name: Optional[str] = Header(None, alias="X-User-Name")
):
    """Process a query with authentication and session management"""
    if rag_system is None or session_manager is None:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    logger.info("="*80)
    logger.info("NEW QUERY REQUEST RECEIVED")
    logger.info("="*80)
    logger.info(f"Query: '{request_body.query}'")
    logger.info(f"User: {x_user_name if x_user_name else 'UNKNOWN'}")
    logger.info(f"Max Tokens: {request_body.max_tokens}, Temperature: {request_body.temperature}, Top-P: {request_body.top_p}")
    
    # Log received headers
    logger.info("-"*80)
    logger.info("HEADERS RECEIVED:")
    logger.info(f"  X-Client-ID: {x_client_id if x_client_id else 'MISSING'} (informational only)")
    logger.info(f"  X-Company-Pin: {x_company_pin if x_company_pin else 'MISSING'} (used for auth)")
    logger.info(f"  X-API-Key: {x_api_key[:20] + '...' if x_api_key else 'MISSING'} (used for auth)")
    logger.info(f"  X-User-Name: {x_user_name if x_user_name else 'MISSING'} (informational only)")
    logger.info("-"*80)
    
    # Validate required headers (only company_pin and api_key needed for auth)
    if not x_company_pin or not x_api_key:
        logger.warning("❌ AUTHENTICATION FAILED: Missing required authentication headers (X-Company-Pin or X-API-Key)")
        logger.info("="*80)
        return QueryResponse(response="Unauthorized", session_id="")
    
    try:
        # Authenticate client against PostgreSQL
        logger.info("🔐 AUTHENTICATING CLIENT...")
        from database.postgres_manager import PostgresManager
        db = PostgresManager(POSTGRES_CONNECTION_STRING)
        
        if not db.connect():
            logger.error("❌ Failed to connect to PostgreSQL database")
            logger.info("="*80)
            return QueryResponse(response="Unauthorized", session_id="")
        
        logger.info("✅ Connected to PostgreSQL database")
        
        # Validate credentials (only company_pin and api_key)
        logger.info(f"🔍 Validating credentials (Company PIN + API Key)")
        client = db.authenticate_client(x_company_pin, x_api_key)
        
        if not client:
            logger.warning(f"❌ AUTHENTICATION FAILED")
            logger.warning("   Reason: Invalid company_pin or api_key combination")
            db.disconnect()
            logger.info("="*80)
            return QueryResponse(response="Unauthorized", session_id="")
        
        # Get client_id from authenticated result
        authenticated_client_id = client['client_id']
        company_pin = client['company_pin']
        
        # Set client context for logging
        set_client_context(company_pin)
        
        logger.info(f"✅ AUTHENTICATION SUCCESSFUL")
        logger.info(f"   Client ID: {authenticated_client_id} (from database)")
        logger.info(f"   Company PIN: {company_pin}")
        logger.info(f"   API Key: {client['api_key'][:20]}...")
        logger.info(f"   User: {x_user_name}")
        logger.info(f"   Active: {client['is_active']}")
        
        # Load client-specific index
        logger.info("-"*80)
        logger.info(f"📂 LOADING CLIENT-SPECIFIC INDEX")
        logger.info(f"Loading embeddings for client: {company_pin}")
        index_loaded = rag_system.load_client_index(company_pin)
        
        if not index_loaded:
            logger.warning(f"⚠️  No index found for client {company_pin}")
            logger.warning("   Client needs to sync documents first")
            db.disconnect()
            clear_client_context()
            logger.info("="*80)
            return QueryResponse(
                response="No documents have been synced for your account yet. Please sync your HR policy documents first.",
                session_id=""
            )
        
        logger.info(f"✅ Client index loaded successfully")
        logger.info(f"   Vectors: {rag_system.vector_store.index.ntotal}")
        logger.info(f"   Chunks: {len(rag_system.vector_store.chunks)}")
        
        # Check token limit
        logger.info("-"*80)
        logger.info("🔢 TOKEN LIMIT CHECK")
        token_status = db.check_token_limit(authenticated_client_id)
        logger.info(f"   Token Limit: {token_status['limit']:,}")
        logger.info(f"   Tokens Used: {token_status['usage']:,}")
        logger.info(f"   Remaining: {token_status['remaining']:,}")
        logger.info(f"   Allowed: {token_status['allowed']}")
        
        if not token_status['allowed']:
            logger.warning(f"❌ TOKEN LIMIT EXCEEDED")
            logger.warning(f"   Client has used {token_status['usage']:,} / {token_status['limit']:,} tokens this month")
            db.disconnect()
            clear_client_context()
            logger.info("="*80)
            return QueryResponse(
                response="Your monthly token limit has been reached. Please contact your administrator.",
                session_id=""
            )
        
        logger.info(f"✅ Token limit check passed")
        
        # Get or create session
        logger.info("-"*80)
        logger.info("📋 SESSION MANAGEMENT")
        session_id = request.session.get("session_id")
        logger.info(f"Session ID from cookie: {session_id if session_id else 'None (new session)'}")
        
        session = None
        if session_id:
            session = session_manager.get_session(session_id)
            if session:
                logger.info(f"✅ Using existing session: {session.session_id}")
                logger.info(f"   Messages in session: {len(session.messages)}")
        
        if not session:
            # Create new session
            session = session_manager.create_session()
            request.session["session_id"] = session.session_id
            logger.info(f"🆕 Created new in-memory session: {session.session_id}")
            
            # Create session in PostgreSQL
            logger.info(f"💾 Saving session to PostgreSQL database...")
            session_db_id = db.create_session(session.session_id, x_user_name or "unknown", authenticated_client_id)
            if session_db_id:
                # Store session_db_id in session for later use
                session.session_db_id = session_db_id
                logger.info(f"✅ Session saved to database")
                logger.info(f"   Table: chatbot.sessions")
                logger.info(f"   Session DB ID: {session_db_id}")
                logger.info(f"   Session UUID: {session.session_id}")
                logger.info(f"   Username: {x_user_name or 'unknown'}")
                logger.info(f"   Client ID: {authenticated_client_id}")
            else:
                logger.warning(f"⚠️  Failed to save session to database")
        
        # Add user message to session
        logger.info("-"*80)
        logger.info("💬 PROCESSING QUERY")
        from fastapi_session_manager import Message
        user_message = Message(
            role="user",
            content=request_body.query,
            timestamp=session.last_active
        )
        session.messages.append(user_message)
        logger.info(f"Added user message to session (total messages: {len(session.messages)})")
        
        # Get recent context from session
        recent_context = ""
        if len(session.messages) > 1:
            # Get last few exchanges (excluding the current user message)
            recent_messages = session.messages[:-1][-(RECENT_CONTEXT_EXCHANGES * 2):]
            logger.info(f"Using {len(recent_messages)} previous messages as context")
            for msg in recent_messages:
                role = "Human" if msg.role == "user" else "Assistant"
                recent_context += f"{role}: {msg.content}\n"
        else:
            logger.info("No previous context (first message in session)")
        
        # Process query with context
        logger.info("🤖 Generating response using RAG system...")
        response, sources = rag_system.query_with_context(
            request_body.query,
            recent_context=recent_context,
            max_tokens=request_body.max_tokens,
            temperature=request_body.temperature,
            top_p=request_body.top_p
        )
        logger.info(f"✅ Response generated ({len(response)} characters, {len(sources)} source documents)")
        
        # Count tokens used (input + output)
        logger.info("-"*80)
        logger.info("🔢 TOKEN COUNTING")
        input_tokens = utils.count_tokens(request_body.query)
        output_tokens = utils.count_tokens(response)
        total_tokens = input_tokens + output_tokens
        logger.info(f"   Input tokens: {input_tokens:,}")
        logger.info(f"   Output tokens: {output_tokens:,}")
        logger.info(f"   Total tokens: {total_tokens:,}")
        
        # Add assistant response to session
        assistant_message = Message(
            role="assistant",
            content=response,
            timestamp=datetime.now().isoformat(),
            context_docs=[{
                "source_file": doc.source_file,
                "chunk_id": doc.chunk_id,
                "relevance_score": doc.relevance_score
            } for doc in sources]
        )
        session.messages.append(assistant_message)
        logger.info(f"Added assistant response to session (total messages: {len(session.messages)})")
        
        # Update session in memory
        session_manager.update_session(session)
        logger.info("✅ Updated in-memory session")
        
        # Save conversation to PostgreSQL in real-time
        logger.info("-"*80)
        logger.info("💾 SAVING TO DATABASE")
        logger.info(f"Saving conversation to PostgreSQL...")
        
        # Get session_db_id (either from session or query database)
        session_db_id = getattr(session, 'session_db_id', None)
        if not session_db_id:
            session_db_id = db.get_session_db_id(session.session_id)
            if session_db_id:
                session.session_db_id = session_db_id
        
        if session_db_id:
            month_abbr = datetime.now().strftime("%b").lower()
            year = datetime.now().strftime("%Y")
            conv_success = db.add_conversation(session_db_id, request_body.query, response, total_tokens)
            if conv_success:
                logger.info(f"✅ Conversation saved to database")
                logger.info(f"   Table: chatbot.conversation_{month_abbr}_{year}")
                logger.info(f"   Session DB ID: {session_db_id}")
                logger.info(f"   Session UUID: {session.session_id}")
                logger.info(f"   Tokens Used: {total_tokens:,}")
                logger.info(f"   User Message: {request_body.query[:50]}...")
                logger.info(f"   Bot Response: {response[:50]}...")
            else:
                logger.warning(f"⚠️  Failed to save conversation to database")
        else:
            logger.warning(f"⚠️  Could not get session DB ID, conversation not saved")
        
        # Update token usage for client
        logger.info(f"Updating token usage for client {authenticated_client_id}...")
        token_update_success = db.update_token_usage(authenticated_client_id, total_tokens)
        if token_update_success:
            new_usage = db.get_client_token_usage(authenticated_client_id)
            logger.info(f"✅ Token usage updated")
            logger.info(f"   Client ID: {authenticated_client_id}")
            logger.info(f"   Tokens Added: {total_tokens:,}")
            logger.info(f"   New Total Usage: {new_usage:,} / {token_status['limit']:,}")
            logger.info(f"   Remaining: {token_status['limit'] - new_usage:,}")
        else:
            logger.warning(f"⚠️  Failed to update token usage")
        
        # Update session activity timestamp
        logger.info(f"Updating session activity timestamp...")
        activity_success = db.update_session_activity(session.session_id)
        if activity_success:
            logger.info(f"✅ Session activity updated")
        else:
            logger.warning(f"⚠️  Failed to update session activity")
        
        # Disconnect from database
        db.disconnect()
        logger.info("✅ Disconnected from PostgreSQL database")
        
        logger.info("-"*80)
        logger.info("✅ QUERY PROCESSED SUCCESSFULLY")
        logger.info(f"   Session ID: {session.session_id}")
        logger.info(f"   Source Documents: {len(sources)}")
        logger.info(f"   Response Length: {len(response)} characters")
        logger.info("="*80)
        
        # Clear client context
        clear_client_context()
        
        return QueryResponse(response=response, session_id=session.session_id)
    
    except Exception as e:
        logger.error(f"Query processing error: {e}", exc_info=True)
        clear_client_context()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/documents")
async def list_documents():
    """List loaded documents"""
    if rag_system is None:
        raise HTTPException(status_code=503, detail="RAG system not initialized")
    
    docs = [
        {
            "name": doc["name"],
            "type": doc["type"],
            "size": len(doc["content"])
        }
        for doc in rag_system.documents
    ]
    
    return {"documents": docs, "count": len(docs)}


@app.get("/sessions")
async def list_sessions():
    """List all active sessions"""
    if session_manager is None:
        raise HTTPException(status_code=503, detail="Session manager not initialized")
    
    sessions = session_manager.list_sessions()
    return {
        "sessions": sessions,
        "count": len(sessions)
    }


@app.get("/chat")
async def chat(message: str, request: Request):
    """Simple GET endpoint for testing - sends a message and gets response"""
    if rag_system is None or session_manager is None:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    if not message:
        raise HTTPException(status_code=400, detail="Message parameter is required")
    
    logger.info(f"Chat endpoint query: '{message}'")
    
    try:
        # Get or create session
        session_id = request.session.get("session_id")
        session = None
        
        if session_id:
            session = session_manager.get_session(session_id)
        
        if not session:
            session = session_manager.create_session()
            request.session["session_id"] = session.session_id
        
        # Get recent context
        recent_context = ""
        if session.messages:
            recent_messages = session.messages[-(RECENT_CONTEXT_EXCHANGES * 2):]
            for msg in recent_messages:
                role = "Human" if msg.role == "user" else "Assistant"
                recent_context += f"{role}: {msg.content}\n"
        
        # Process query
        response, sources = rag_system.query_with_context(
            message,
            recent_context=recent_context,
            max_tokens=DEFAULT_MAX_TOKENS,
            temperature=DEFAULT_TEMPERATURE,
            top_p=DEFAULT_TOP_P
        )
        
        # Add messages to session
        from fastapi_session_manager import Message
        session.messages.append(Message(
            role="user",
            content=message,
            timestamp=datetime.now().isoformat()
        ))
        session.messages.append(Message(
            role="assistant",
            content=response,
            timestamp=datetime.now().isoformat()
        ))
        session_manager.update_session(session)
        
        return {"response": response, "session_id": session.session_id}
    
    except Exception as e:
        logger.error(f"Chat error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# Admin Endpoints

class SyncClientRequest(BaseModel):
    company_pin: str
    api_key: str
    is_active: bool = True


@app.post("/admin/sync-client")
async def sync_client(request_body: SyncClientRequest):
    """
    Sync client data from HCMSAPI to PostgreSQL
    Called when API key is generated in HCMSAPI
    
    Note: PostgreSQL will auto-assign client_id (1, 2, 3...)
    """
    logger.info(f"Syncing client from HCMSAPI to PostgreSQL (company_pin: {request_body.company_pin})")
    
    try:
        from database.postgres_manager import PostgresManager
        db = PostgresManager(POSTGRES_CONNECTION_STRING)
        
        if not db.connect():
            raise HTTPException(status_code=503, detail="Database connection failed")
        
        success = db.sync_client(
            request_body.company_pin,
            request_body.api_key,
            request_body.is_active
        )
        
        db.disconnect()
        
        if success:
            logger.info(f"Client synced successfully (company_pin: {request_body.company_pin})")
            return {
                "success": True,
                "message": f"Client synced successfully (company_pin: {request_body.company_pin})"
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to sync client")
    
    except Exception as e:
        logger.error(f"Error syncing client: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# Document Sync Endpoints

class DocumentData(BaseModel):
    announcement_id: int
    file_name: str
    document_title: str
    file_content: str  # base64 encoded
    file_extension: str
    created_on: str


class SyncDocumentRequest(BaseModel):
    client_id: int  # Informational only
    company_pin: str  # Used for authentication
    company_name: str  # Informational only
    documents: list[DocumentData]  # Changed from single document to list


@app.post("/documents/sync-batch")
async def sync_single_document(request_body: SyncDocumentRequest):
    """
    Sync HR policy documents from HCMS backend (batch mode)
    
    IMPORTANT: This endpoint rebuilds the entire index from scratch to avoid duplicates
    
    Process:
    1. Authenticate client
    2. Delete existing document records for this client
    3. Delete existing FAISS index
    4. Load all documents from docs/ folder
    5. Process all new documents from API
    6. Combine all documents and build fresh index
    7. Store document metadata in PostgreSQL
    """
    logger.info("="*80)
    logger.info("DOCUMENT SYNC REQUEST RECEIVED (BATCH MODE - FULL REBUILD)")
    logger.info("="*80)
    logger.info(f"Company: {request_body.company_name}")
    logger.info(f"Company PIN: {request_body.company_pin}")
    logger.info(f"Client ID (informational): {request_body.client_id}")
    logger.info(f"Documents to sync: {len(request_body.documents)}")
    for idx, doc in enumerate(request_body.documents, 1):
        logger.info(f"  [{idx}] {doc.file_name} ({doc.file_extension}) - {doc.document_title}")
    logger.info("⚠️  NOTE: This will rebuild the entire index from scratch")
    
    if rag_system is None:
        logger.error("❌ RAG system not initialized")
        raise HTTPException(status_code=503, detail="RAG system not initialized")
    
    try:
        # Set client context for logging
        set_client_context(request_body.company_pin)
        
        # Authenticate client
        logger.info("-"*80)
        logger.info("🔐 AUTHENTICATING CLIENT")
        from database.postgres_manager import PostgresManager
        db = PostgresManager(POSTGRES_CONNECTION_STRING)
        
        if not db.connect():
            logger.error("❌ Failed to connect to PostgreSQL")
            clear_client_context()
            raise HTTPException(status_code=503, detail="Database connection failed")
        
        logger.info("✅ Connected to PostgreSQL")
        
        # Get client by company_pin only (no API key needed for document sync)
        from psycopg2.extras import RealDictCursor
        with db.conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                "SELECT client_id, company_pin, is_active FROM chatbot.clients WHERE company_pin = %s",
                (request_body.company_pin,)
            )
            client = cur.fetchone()
        
        if not client:
            logger.warning(f"❌ Client not found for company_pin: {request_body.company_pin}")
            db.disconnect()
            clear_client_context()
            raise HTTPException(status_code=404, detail="Client not found")
        
        if not client['is_active']:
            logger.warning(f"❌ Client is inactive: {request_body.company_pin}")
            db.disconnect()
            clear_client_context()
            raise HTTPException(status_code=403, detail="Client is inactive")
        
        authenticated_client_id = client['client_id']
        company_pin = client['company_pin']
        logger.info(f"✅ Client authenticated (client_id: {authenticated_client_id}, company_pin: {company_pin})")
        
        # Set client-specific paths for embeddings
        logger.info("-"*80)
        logger.info("📁 SETTING CLIENT-SPECIFIC PATHS")
        rag_system.vector_store.set_client_paths(company_pin)
        logger.info(f"✅ Client paths set for: {company_pin}")
        
        # Delete all existing document records for this client
        logger.info("-"*80)
        logger.info("🗑️  CLEARING EXISTING DOCUMENT RECORDS")
        logger.info(f"Deleting all document records for client_id: {authenticated_client_id}")
        delete_success = db.delete_client_documents(authenticated_client_id)
        if delete_success:
            logger.info("✅ Existing document records cleared")
        else:
            logger.warning("⚠️  Failed to clear document records (continuing anyway)")
        
        # Process all API documents
        logger.info("-"*80)
        logger.info(f"📄 PROCESSING {len(request_body.documents)} API DOCUMENTS")
        from processing.document_processor import DocumentProcessor
        processor = DocumentProcessor(DOCS_FOLDER)
        
        api_documents = []
        for idx, doc in enumerate(request_body.documents, 1):
            logger.info(f"Processing document [{idx}/{len(request_body.documents)}]: {doc.file_name}")
            
            text_content = processor.process_base64_document(
                doc.file_content,
                doc.file_name,
                doc.file_extension
            )
            
            if not text_content or not text_content.strip():
                logger.warning(f"⚠️  Failed to extract text from {doc.file_name}, skipping...")
                continue
            
            logger.info(f"✅ Extracted {len(text_content)} characters from {doc.file_name}")
            
            # Add to API documents list
            api_documents.append({
                'name': doc.file_name,
                'content': text_content,
                'type': doc.file_extension,
                'path': f'api_sync/{doc.file_name}',
                'announcement_id': doc.announcement_id,
                'document_title': doc.document_title
            })
        
        logger.info(f"✅ Successfully processed {len(api_documents)} out of {len(request_body.documents)} documents")
        
        # Load all documents from docs/ folder
        logger.info("-"*80)
        logger.info("📂 LOADING DOCUMENTS FROM DOCS FOLDER")
        logger.info(f"Loading existing documents from: {DOCS_FOLDER}")
        folder_documents = rag_system.doc_processor.load_documents()
        logger.info(f"✅ Loaded {len(folder_documents)} document(s) from folder")
        
        # Combine folder documents + API documents
        logger.info("-"*80)
        logger.info("📄 COMBINING DOCUMENTS")
        all_documents = folder_documents + api_documents
        logger.info(f"✅ Total documents to process: {len(all_documents)}")
        logger.info(f"   From folder: {len(folder_documents)}")
        logger.info(f"   From API: {len(api_documents)}")
        
        # Chunk all documents
        logger.info("-"*80)
        logger.info("✂️  CHUNKING ALL DOCUMENTS")
        from processing.chunking import SemanticChunker
        chunker = SemanticChunker(similarity_threshold=SEMANTIC_SIMILARITY_THRESHOLD)
        
        all_chunks = chunker.create_chunks(all_documents)
        logger.info(f"✅ Created {len(all_chunks)} total chunks from all documents")
        
        if not all_chunks:
            logger.error("❌ No chunks created from documents")
            db.disconnect()
            clear_client_context()
            raise HTTPException(status_code=400, detail="Failed to create chunks from documents")
        
        # Rebuild vector store from scratch
        logger.info("-"*80)
        logger.info("🔨 REBUILDING VECTOR STORE FROM SCRATCH")
        logger.info(f"Rebuilding index with {len(all_chunks)} chunks...")
        
        success = rag_system.vector_store.rebuild_index_from_scratch(all_chunks)
        
        if not success:
            logger.error("❌ Failed to rebuild vector store")
            db.disconnect()
            clear_client_context()
            raise HTTPException(status_code=500, detail="Failed to rebuild vector store")
        
        logger.info(f"✅ Vector store rebuilt successfully")
        logger.info(f"   Total vectors in index: {rag_system.vector_store.index.ntotal}")
        logger.info(f"   Total chunks: {len(rag_system.vector_store.chunks)}")
        
        # Create/update retriever with new chunks
        logger.info("🔄 Creating retriever with new chunks...")
        from retrieval.retriever import SemanticRetriever
        rag_system.retriever = SemanticRetriever(rag_system.vector_store, rag_system.vector_store.chunks)
        logger.info("✅ Retriever created")
        
        # Store metadata for all API documents (not folder documents)
        logger.info("-"*80)
        logger.info("💾 SAVING API DOCUMENTS METADATA")
        logger.info(f"Saving metadata for {len(api_documents)} API document(s)...")
        logger.info(f"Note: Folder documents ({len(folder_documents)}) are indexed but not tracked in database")
        
        saved_count = 0
        for idx, api_doc in enumerate(api_documents, 1):
            # Count chunks for this document
            doc_chunks = [c for c in all_chunks if c.source_file == api_doc['name']]
            
            document_id = db.add_document(
                announcement_id=api_doc['announcement_id'],
                client_id=authenticated_client_id,
                file_name=api_doc['name'],
                document_title=api_doc['document_title'],
                file_extension=api_doc['type'],
                chunk_count=len(doc_chunks)
            )
            
            if document_id:
                logger.info(f"  ✅ [{idx}/{len(api_documents)}] {api_doc['name']} (document_id: {document_id}, chunks: {len(doc_chunks)})")
                saved_count += 1
            else:
                logger.warning(f"  ⚠️  [{idx}/{len(api_documents)}] Failed to save {api_doc['name']}")
        
        logger.info(f"✅ Saved {saved_count}/{len(api_documents)} API documents to database")
        
        db.disconnect()
        logger.info("✅ Disconnected from PostgreSQL")
        
        logger.info("-"*80)
        logger.info("✅ DOCUMENT SYNC COMPLETED SUCCESSFULLY")
        logger.info(f"   Client ID: {authenticated_client_id}")
        logger.info(f"   Total Documents Processed: {len(all_documents)}")
        logger.info(f"   Folder Documents: {len(folder_documents)}")
        logger.info(f"   API Documents: {len(api_documents)}")
        logger.info(f"   Total Chunks Created: {len(all_chunks)}")
        logger.info(f"   Documents Saved to DB: {saved_count}")
        logger.info(f"   Index Rebuilt: Yes (from scratch)")
        logger.info("="*80)
        
        clear_client_context()
        
        return {
            "success": True,
            "message": f"Successfully synced {len(api_documents)} document(s) (index rebuilt from scratch)",
            "total_documents": len(all_documents),
            "folder_documents": len(folder_documents),
            "api_documents": len(api_documents),
            "documents_saved": saved_count,
            "chunks_created": len(all_chunks),
            "total_vectors": rag_system.vector_store.index.ntotal,
            "rebuild_mode": True
        }
    
    except HTTPException:
        clear_client_context()
        raise
    except Exception as e:
        logger.error(f"❌ Error syncing document: {e}", exc_info=True)
        clear_client_context()
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    logger.info(f"Starting server on {API_HOST}:{API_PORT}")
    uvicorn.run(
        app,
        host=API_HOST,
        port=API_PORT,
        reload=False,
        log_level="info"
    )
