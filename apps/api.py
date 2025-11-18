"""FastAPI server for the RAG chatbot"""
import logging
import os
import sys
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import uvicorn

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from config import *
from rag_system import RAGSystem
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

# Global RAG system
rag_system: Optional[RAGSystem] = None


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
    sources: List[SourceDocument]
    conversation_id: Optional[str] = None


class SystemStatus(BaseModel):
    status: str
    documents_loaded: int
    chunks_created: int
    model_loaded: bool


@app.on_event("startup")
async def startup_event():
    """Initialize RAG system on startup"""
    global rag_system
    logger.info("="*60)
    logger.info("Starting FastAPI server")
    logger.info("="*60)
    
    try:
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
    if rag_system is None:
        raise HTTPException(status_code=503, detail="RAG system not initialized")
    
    return SystemStatus(
        status="healthy",
        documents_loaded=len(rag_system.documents),
        chunks_created=len(rag_system.chunks),
        model_loaded=rag_system.llm_engine.model is not None
    )


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """Process a query"""
    if rag_system is None:
        raise HTTPException(status_code=503, detail="RAG system not initialized")
    
    logger.info(f"API query received: '{request.query}'")
    
    try:
        response, sources = rag_system.query(
            request.query,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p
        )
        
        source_docs = [
            SourceDocument(
                content=doc.content,
                source_file=doc.source_file,
                chunk_id=doc.chunk_id,
                relevance_score=doc.relevance_score
            )
            for doc in sources
        ]
        
        logger.info(f"API query processed successfully")
        
        return QueryResponse(
            response=response,
            sources=source_docs
        )
    
    except Exception as e:
        logger.error(f"Query processing error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/clear")
async def clear_conversation():
    """Clear conversation history"""
    if rag_system is None:
        raise HTTPException(status_code=503, detail="RAG system not initialized")
    
    rag_system.clear_conversation()
    logger.info("Conversation cleared via API")
    
    return {"message": "Conversation history cleared"}


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


@app.get("/chat")
async def chat(message: str):
    """Simple GET endpoint for testing - sends a message and gets response"""
    if rag_system is None:
        raise HTTPException(status_code=503, detail="RAG system not initialized")
    
    if not message:
        raise HTTPException(status_code=400, detail="Message parameter is required")
    
    logger.info(f"Chat endpoint query: '{message}'")
    
    try:
        response, sources = rag_system.query(
            message,
            max_tokens=DEFAULT_MAX_TOKENS,
            temperature=DEFAULT_TEMPERATURE,
            top_p=DEFAULT_TOP_P
        )
        
        return {
            "message": message,
            "response": response,
            "sources_count": len(sources)
        }
    
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
