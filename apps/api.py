"""FastAPI server for the RAG chatbot"""
import logging
import os
import sys
from datetime import datetime
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.sessions import SessionMiddleware
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
import secrets

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from config import *
from rag_system import RAGSystem
from fastapi_session_manager import FastAPISessionManager
import utils

# Setup logging
utils.setup_logging(LOG_LEVEL)
logger = logging.getLogger(__name__)

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
        
        # Initialize RAG system
        rag_system = RAGSystem()
        success = rag_system.initialize()
        
        if success:
            logger.info("RAG system initialized successfully")
        else:
            logger.error("Failed to initialize RAG system")
            raise Exception("RAG system initialization failed")
        
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
async def query(request_body: QueryRequest, request: Request):
    """Process a query with session management"""
    if rag_system is None or session_manager is None:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    logger.info(f"API query received: '{request_body.query}'")
    
    try:
        # Get or create session
        session_id = request.session.get("session_id")
        logger.debug(f"Session ID from cookie: {session_id}")
        
        session = None
        if session_id:
            session = session_manager.get_session(session_id)
            if session:
                logger.debug(f"Using existing session: {session.session_id}")
        
        if not session:
            # Create new session
            session = session_manager.create_session()
            request.session["session_id"] = session.session_id
            logger.info(f"Created new session: {session.session_id}")
        
        # Add user message to session
        from fastapi_session_manager import Message
        user_message = Message(
            role="user",
            content=request_body.query,
            timestamp=session.last_active
        )
        session.messages.append(user_message)
        
        # Get recent context from session
        recent_context = ""
        if len(session.messages) > 1:
            # Get last few exchanges (excluding the current user message)
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
        
        # Update session
        session_manager.update_session(session)
        
        logger.info(f"Query processed successfully (session: {session.session_id}, sources: {len(sources)})")
        
        return QueryResponse(response=response, session_id=session.session_id)
    
    except Exception as e:
        logger.error(f"Query processing error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/clear")
async def clear_conversation(request: Request):
    """Clear conversation history for current session"""
    if session_manager is None:
        raise HTTPException(status_code=503, detail="Session manager not initialized")
    
    # Get session
    session_id = request.session.get("session_id")
    if not session_id:
        raise HTTPException(status_code=400, detail="No active session")
    
    # Clear session messages
    success = session_manager.clear_session(session_id)
    
    if success:
        logger.info(f"Cleared conversation for session: {session_id}")
        return {
            "message": "Conversation history cleared",
            "session_id": session_id
        }
    else:
        raise HTTPException(status_code=404, detail="Session not found")


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


if __name__ == "__main__":
    logger.info(f"Starting server on {API_HOST}:{API_PORT}")
    uvicorn.run(
        app,
        host=API_HOST,
        port=API_PORT,
        reload=False,
        log_level="info"
    )
