"""FastAPI server for the RAG chatbot"""
import asyncio
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
from psycopg2.extras import RealDictCursor

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from config import (
    SECRET_KEY, SESSION_MAX_AGE, MAX_SESSIONS, API_HOST, API_PORT,
    CLEANUP_INTERVAL, POSTGRES_CONNECTION_STRING, LOG_LEVEL,
    DEFAULT_MAX_TOKENS, DEFAULT_TEMPERATURE, DEFAULT_TOP_P,
    RECENT_CONTEXT_EXCHANGES, DOCS_FOLDER, SEMANTIC_SIMILARITY_THRESHOLD
)
from rag_system import RAGSystem
from fastapi_session_manager import FastAPISessionManager, Message
from database.postgres_manager import PostgresManager
from processing.document_processor import DocumentProcessor
from processing.chunking import SemanticChunker
from retrieval.retriever import SemanticRetriever
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
    logger.info("Starting FastAPI server")
    
    try:
        session_manager = FastAPISessionManager(
            max_sessions=MAX_SESSIONS,
            session_max_age=SESSION_MAX_AGE
        )
        rag_system = RAGSystem()
        asyncio.create_task(cleanup_sessions_task())
        logger.info("Server initialized successfully")
    
    except Exception as e:
        logger.error(f"Startup error: {e}", exc_info=True)
        raise


async def cleanup_sessions_task():
    """Background task to clean up expired sessions"""
    while True:
        await asyncio.sleep(CLEANUP_INTERVAL)
        try:
            if session_manager:
                session_manager.cleanup_expired_sessions()
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
    
    logger.info(f"Query received from user: {x_user_name or 'UNKNOWN'}")
    
    # Validate required headers (only company_pin and api_key needed for auth)
    if not x_company_pin or not x_api_key:
        logger.warning("Authentication failed: Missing credentials")
        return QueryResponse(response="Unauthorized", session_id="")
    
    try:
        # Authenticate client against PostgreSQL
        db = PostgresManager(POSTGRES_CONNECTION_STRING)
        
        if not db.connect():
            logger.error("Failed to connect to database")
            return QueryResponse(response="Unauthorized", session_id="")
        
        client = db.authenticate_client(x_company_pin, x_api_key)
        
        if not client:
            logger.warning("Authentication failed: Invalid credentials")
            db.disconnect()
            return QueryResponse(response="Unauthorized", session_id="")
        
        authenticated_client_id = client['client_id']
        company_pin = client['company_pin']
        set_client_context(company_pin)
        
        logger.info(f"Client authenticated: {company_pin} (user: {x_user_name})")
        
        # Load client-specific HR policy index (if exists)
        # Note: General index is already loaded during RAG system initialization
        rag_system.load_client_index(company_pin)
        
        # Check token limit
        token_status = db.check_token_limit(authenticated_client_id)
        
        if not token_status['allowed']:
            logger.warning(f"Token limit exceeded for client {authenticated_client_id}")
            db.disconnect()
            clear_client_context()
            return QueryResponse(
                response="Your monthly token limit has been reached. Please contact your administrator.",
                session_id=""
            )
        
        # Get or create session
        session_id = request.session.get("session_id")
        session = None
        if session_id:
            session = session_manager.get_session(session_id)
        
        if not session:
            session = session_manager.create_session()
            request.session["session_id"] = session.session_id
            session_db_id = db.create_session(session.session_id, x_user_name or "unknown", authenticated_client_id)
            if session_db_id:
                session.session_db_id = session_db_id
                logger.info(f"New session created: {session.session_id}")
        
        # Add user message to session
        user_message = Message(
            role="user",
            content=request_body.query,
            timestamp=session.last_active
        )
        session.messages.append(user_message)
        
        # Get recent context from session
        recent_context = ""
        if len(session.messages) > 1:
            recent_messages = session.messages[:-1][-(RECENT_CONTEXT_EXCHANGES * 2):]
            for msg in recent_messages:
                role = "Human" if msg.role == "user" else "Assistant"
                recent_context += f"{role}: {msg.content}\n"
        
        # Process query with context
        response, sources = rag_system.query_with_context(
            request_body.query,
            recent_context=recent_context,
            max_tokens=request_body.max_tokens,
            temperature=request_body.temperature,
            top_p=request_body.top_p
        )
        
        # Count tokens used (input + output)
        input_tokens = utils.count_tokens(request_body.query)
        output_tokens = utils.count_tokens(response)
        total_tokens = input_tokens + output_tokens
        
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
        session_manager.update_session(session)
        
        # Save conversation to PostgreSQL
        session_db_id = getattr(session, 'session_db_id', None)
        if not session_db_id:
            session_db_id = db.get_session_db_id(session.session_id)
            if session_db_id:
                session.session_db_id = session_db_id
        
        if session_db_id:
            db.add_conversation(session_db_id, request_body.query, response, total_tokens)
        
        # Update token usage and session activity
        db.update_token_usage(authenticated_client_id, total_tokens)
        db.update_session_activity(session.session_id)
        db.disconnect()
        
        logger.info(f"Query processed: {total_tokens} tokens, {len(sources)} sources")
        
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
    try:
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
            logger.info(f"Client synced: {request_body.company_pin}")
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
    logger.info(f"Document sync request: {request_body.company_name} ({len(request_body.documents)} documents)")
    
    if rag_system is None:
        logger.error("RAG system not initialized")
        raise HTTPException(status_code=503, detail="RAG system not initialized")
    
    try:
        # Set client context for logging
        set_client_context(request_body.company_pin)
        
        db = PostgresManager(POSTGRES_CONNECTION_STRING)
        
        if not db.connect():
            logger.error("Failed to connect to database")
            clear_client_context()
            raise HTTPException(status_code=503, detail="Database connection failed")
        
        # Get client by company_pin only (no API key needed for document sync)
        with db.conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                "SELECT client_id, company_pin, is_active FROM chatbot.clients WHERE company_pin = %s",
                (request_body.company_pin,)
            )
            client = cur.fetchone()
        
        if not client:
            logger.warning(f"Client not found: {request_body.company_pin}")
            db.disconnect()
            clear_client_context()
            raise HTTPException(status_code=404, detail="Client not found")
        
        if not client['is_active']:
            logger.warning(f"Client is inactive: {request_body.company_pin}")
            db.disconnect()
            clear_client_context()
            raise HTTPException(status_code=403, detail="Client is inactive")
        
        authenticated_client_id = client['client_id']
        company_pin = client['company_pin']
        
        # Set client-specific paths for embeddings
        rag_system.vector_store.set_client_paths(company_pin)
        
        # Delete all existing document records for this client
        delete_success = db.delete_client_documents(authenticated_client_id)
        if not delete_success:
            logger.warning("Failed to clear document records")
        
        # Process all API documents
        processor = DocumentProcessor(DOCS_FOLDER)
        
        api_documents = []
        for idx, doc in enumerate(request_body.documents, 1):
            text_content = processor.process_base64_document(
                doc.file_content,
                doc.file_name,
                doc.file_extension
            )
            
            if not text_content or not text_content.strip():
                logger.warning(f"Failed to extract text from {doc.file_name}")
                continue
            
            # Add to API documents list
            api_documents.append({
                'name': doc.file_name,
                'content': text_content,
                'type': doc.file_extension,
                'path': f'api_sync/{doc.file_name}',
                'announcement_id': doc.announcement_id,
                'document_title': doc.document_title
            
            })
        

        
        # Chunk only API documents (HR policies)
        chunker = SemanticChunker(similarity_threshold=SEMANTIC_SIMILARITY_THRESHOLD)
        
        hr_policy_chunks = chunker.create_chunks(api_documents)
        
        if not hr_policy_chunks:
            logger.error("No chunks created from HR policy documents")
            db.disconnect()
            clear_client_context()
            raise HTTPException(status_code=400, detail="Failed to create chunks from HR policy documents")
        
        # Rebuild client-specific vector store from scratch (HR policies only)
        success = rag_system.vector_store.rebuild_client_index_from_scratch(hr_policy_chunks)
        
        if not success:
            logger.error("Failed to rebuild client vector store")
            db.disconnect()
            clear_client_context()
            raise HTTPException(status_code=500, detail="Failed to rebuild client vector store")
        
        # Store metadata for HR policy documents
        saved_count = 0
        for idx, api_doc in enumerate(api_documents, 1):
            # Count chunks for this document
            doc_chunks = [c for c in hr_policy_chunks if c.source_file == api_doc['name']]
            
            document_id = db.add_document(
                announcement_id=api_doc['announcement_id'],
                client_id=authenticated_client_id,
                file_name=api_doc['name'],
                document_title=api_doc['document_title'],
                file_extension=api_doc['type'],
                chunk_count=len(doc_chunks)
            )
            
            if document_id:
                saved_count += 1
            else:
                logger.warning(f"Failed to save {api_doc['name']}")
        
        db.disconnect()
        
        logger.info(f"Document sync completed: {len(api_documents)} HR policy docs, {len(hr_policy_chunks)} chunks")
        
        clear_client_context()
        
        return {
            "success": True,
            "message": f"Successfully synced {len(api_documents)} document(s) (index rebuilt from scratch)",
            "api_documents": len(api_documents),
            "documents_saved": saved_count,
            "chunks_created": len(hr_policy_chunks),
            "total_vectors": rag_system.vector_store.client_index.ntotal if rag_system.vector_store.client_index else 0,
            "rebuild_mode": True
        }
    
    except HTTPException:
        clear_client_context()
        raise
    except Exception as e:
        logger.error(f"Error syncing document: {e}", exc_info=True)
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
