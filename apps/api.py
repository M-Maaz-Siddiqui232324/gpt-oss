"""FastAPI server for the RAG chatbot"""
import logging
import os
import sys
import uuid
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import uvicorn

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from config import *
from rag_system import RAGSystem
from session_manager import SessionManager
import utils

# Setup logging
utils.setup_logging(LOG_LEVEL)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(
    title="FlowHCM RAG Chatbot API",
    description="Local RAG chatbot with GPT-OSS-20B",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global RAG system and session manager
rag_system: Optional[RAGSystem] = None
session_manager: Optional[SessionManager] = None


# Request/Response models
class QueryRequest(BaseModel):
    session_id: Optional[str] = None  # Optional, will be generated if not provided
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
    session_id: str
    response: str
    sources_count: int


class SystemStatus(BaseModel):
    status: str
    documents_loaded: int
    chunks_created: int
    model_loaded: bool
    redis_connected: bool
    active_sessions: int


@app.on_event("startup")
async def startup_event():
    """Initialize RAG system and session manager on startup"""
    global rag_system, session_manager
    logger.info("="*60)
    logger.info("Starting FastAPI server")
    logger.info("="*60)
    
    try:
        # Initialize session manager
        logger.info("Initializing Redis session manager")
        session_manager = SessionManager()
        logger.info(f"Active sessions: {session_manager.get_session_count()}")
        
        # Initialize RAG system
        rag_system = RAGSystem()
        success = rag_system.initialize()
        
        if success:
            logger.info("RAG system initialized successfully")
        else:
            logger.error("Failed to initialize RAG system")
            raise Exception("RAG system initialization failed")
    
    except Exception as e:
        logger.error(f"Startup error: {e}", exc_info=True)
        raise


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
    if rag_system is None or session_manager is None:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    redis_connected = True
    active_sessions = 0
    try:
        active_sessions = session_manager.get_session_count()
    except:
        redis_connected = False
    
    return SystemStatus(
        status="healthy",
        documents_loaded=len(rag_system.documents),
        chunks_created=len(rag_system.chunks),
        model_loaded=rag_system.llm_engine.model is not None,
        redis_connected=redis_connected,
        active_sessions=active_sessions
    )


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """Process a query with session management"""
    if rag_system is None or session_manager is None:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    # Generate or use provided session ID
    session_id = request.session_id or str(uuid.uuid4())
    
    logger.info(f"API query received: '{request.query}' [Session: {session_id}]")
    
    try:
        # Get session history for context
        recent_context = session_manager.get_recent_context(session_id, RECENT_CONTEXT_EXCHANGES)
        
        # Process query with RAG system
        response, sources = rag_system.query_with_context(
            request.query,
            recent_context,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p
        )
        
        # Convert sources to serializable format
        sources_data = [
            {
                "content": doc.content,
                "source_file": doc.source_file,
                "chunk_id": doc.chunk_id,
                "relevance_score": doc.relevance_score
            }
            for doc in sources
        ]
        
        # Save to session
        session_manager.add_message(session_id, "user", request.query, sources_data)
        session_manager.add_message(session_id, "assistant", response)
        
        logger.info(f"Query processed [Session: {session_id}] (used {len(sources)} sources)")
        
        return QueryResponse(
            session_id=session_id,
            response=response,
            sources_count=len(sources)
        )
    
    except Exception as e:
        logger.error(f"Query processing error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/clear/{session_id}")
async def clear_conversation(session_id: str):
    """Clear conversation history for a specific session"""
    if session_manager is None:
        raise HTTPException(status_code=503, detail="Session manager not initialized")
    
    if not session_manager.session_exists(session_id):
        raise HTTPException(status_code=404, detail="Session not found")
    
    session_manager.clear_session(session_id)
    logger.info(f"Conversation cleared for session: {session_id}")
    
    return {"message": "Conversation history cleared", "session_id": session_id}


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
    
    active_sessions = session_manager.get_active_sessions()
    return {
        "active_sessions": active_sessions,
        "count": len(active_sessions)
    }


@app.get("/session/{session_id}")
async def get_session_info(session_id: str):
    """Get session information and message history"""
    if session_manager is None:
        raise HTTPException(status_code=503, detail="Session manager not initialized")
    
    session_data = session_manager.get_session(session_id)
    if not session_data:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return session_data


@app.post("/session/{session_id}/end")
async def end_session(session_id: str):
    """End a session - archives to file and removes from Redis"""
    if session_manager is None:
        raise HTTPException(status_code=503, detail="Session manager not initialized")
    
    if not session_manager.session_exists(session_id):
        raise HTTPException(status_code=404, detail="Session not found")
    
    if session_manager.end_session(session_id):
        return {
            "message": "Session ended and archived",
            "session_id": session_id,
            "archived": True
        }
    else:
        raise HTTPException(status_code=500, detail="Failed to end session")


@app.delete("/session/{session_id}")
async def delete_session(session_id: str, archive: bool = True):
    """Delete a session (archives by default)"""
    if session_manager is None:
        raise HTTPException(status_code=503, detail="Session manager not initialized")
    
    if session_manager.delete_session(session_id, archive=archive):
        return {
            "message": "Session deleted",
            "session_id": session_id,
            "archived": archive
        }
    else:
        raise HTTPException(status_code=404, detail="Session not found")


@app.get("/archives")
async def list_archives():
    """List all archived session files"""
    if session_manager is None:
        raise HTTPException(status_code=503, detail="Session manager not initialized")
    
    try:
        archive_files = [
            f for f in os.listdir(session_manager.archive_folder)
            if f.startswith("session_") and f.endswith(".json")
        ]
        archive_files.sort(reverse=True)  # Most recent first
        
        return {
            "archives": archive_files,
            "count": len(archive_files),
            "location": session_manager.archive_folder
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/archives/{filename}")
async def get_archive(filename: str):
    """Get a specific archived session"""
    if session_manager is None:
        raise HTTPException(status_code=503, detail="Session manager not initialized")
    
    filepath = os.path.join(session_manager.archive_folder, filename)
    
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="Archive not found")
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except Exception as e:
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
