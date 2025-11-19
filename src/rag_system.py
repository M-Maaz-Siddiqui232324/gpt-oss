"""Main RAG system orchestrating all components"""
import logging
import numpy as np
from typing import List, Tuple
from datetime import datetime

from config import *
from processing.document_processor import DocumentProcessor
from processing.chunking import SemanticChunker, DocumentChunk
from retrieval.vector_store import VectorStore
from retrieval.retriever import SemanticRetriever
from generation.llm_engine import LLMEngine
from generation.prompts import (
    get_general_prompt, 
    get_document_aware_prompt
)
import utils

logger = logging.getLogger(__name__)


class RAGSystem:
    """Complete RAG system"""
    
    def __init__(self):
        logger.info("="*60)
        logger.info("Initializing RAG System")
        logger.info("="*60)
        
        # Components
        self.doc_processor = DocumentProcessor(DOCS_FOLDER)
        self.chunker = SemanticChunker(similarity_threshold=SEMANTIC_SIMILARITY_THRESHOLD)
        self.vector_store = VectorStore(EMBEDDING_MODEL, FAISS_INDEX_FILE, CHUNKS_FILE)
        self.llm_engine = LLMEngine(MODEL_NAME, OLLAMA_BASE_URL)
        self.retriever = None
        
        # Data
        self.documents = []
        self.chunks = []
        self.messages = []
        
        logger.info("RAG System initialized")
    
    def initialize(self) -> bool:
        """Initialize all components"""
        logger.info("Starting system initialization")
        
        # Load LLM
        logger.info("Loading LLM model")
        if not self.llm_engine.load_model():
            logger.error("Failed to load LLM model")
            return False
        logger.info("LLM model loaded successfully")
        
        # Load documents
        logger.info("Loading documents")
        self.documents = self.doc_processor.load_documents()
        if not self.documents:
            logger.warning("No documents loaded")
            return False
        
        # Create chunks
        logger.info("Creating document chunks")
        self.chunks = self.chunker.create_chunks(self.documents)
        if not self.chunks:
            logger.error("Failed to create chunks")
            return False
        
        # Build vector index
        logger.info("Building vector index")
        if not self.vector_store.build_index(self.chunks):
            logger.error("Failed to build vector index")
            return False
        
        # Initialize retriever
        self.retriever = SemanticRetriever(
            self.vector_store, 
            self.chunks
        )
        logger.info("Retriever initialized")
        
        logger.info("="*60)
        logger.info("RAG System initialization complete")
        logger.info(f"Documents: {len(self.documents)}")
        logger.info(f"Chunks: {len(self.chunks)}")
        logger.info("="*60)
        return True
    
    def query(
        self, 
        user_input: str,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
        top_p: float = DEFAULT_TOP_P
    ) -> Tuple[str, List[DocumentChunk]]:
        """Process a user query and generate response (with internal history)"""
        recent_context = self.get_recent_context(RECENT_CONTEXT_EXCHANGES)
        response, sources = self.query_with_context(
            user_input,
            recent_context,
            max_tokens,
            temperature,
            top_p
        )
        
        # Save to internal history
        self.add_message("user", user_input, sources)
        self.add_message("assistant", response)
        
        return response, sources
    
    def query_with_context(
        self, 
        user_input: str,
        recent_context: str = "",
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
        top_p: float = DEFAULT_TOP_P
    ) -> Tuple[str, List[DocumentChunk]]:
        """Process a user query with external context (for session management)"""
        logger.info("="*60)
        logger.info(f"Processing query: '{user_input}'")
        logger.info(f"Params: max_tokens={max_tokens}, temp={temperature}, top_p={top_p}")
        
        try:
            # Retrieve relevant documents
            logger.info("Retrieving relevant documents")
            context_docs = self.retriever.retrieve(user_input, TOP_K_RETRIEVAL)
            
            if not context_docs:
                logger.warning("No context documents found")
                logger.info(">>> DECISION: Using GENERAL RESPONSE (no context docs)")
                return self._generate_general_response(user_input, recent_context, max_tokens, temperature, top_p), []
            
            # Dynamic threshold filtering
            scores = [doc.relevance_score for doc in context_docs]
            mean_score = np.mean(scores)
            std_score = np.std(scores)
            logger.info(f"Score stats: mean={mean_score:.3f}, std={std_score:.3f}")
            
            dynamic_threshold = max(MIN_RELEVANCE_THRESHOLD, mean_score - 0.5 * std_score)
            logger.info(f"Dynamic threshold: {dynamic_threshold:.3f}")
            
            # Filter relevant docs
            relevant_docs = [doc for doc in context_docs if doc.relevance_score >= dynamic_threshold]
            logger.info(f"Filtered to {len(relevant_docs)} relevant documents")
            
            # Take top documents for context
            relevant_docs = relevant_docs[:TOP_K_CONTEXT]
            logger.info(f"Using top {len(relevant_docs)} documents")
            
            if not relevant_docs:
                logger.warning("No documents passed threshold")
                logger.info(">>> DECISION: Using GENERAL RESPONSE (no relevant docs)")
                return self._generate_general_response(user_input, recent_context, max_tokens, temperature, top_p), []
            
            # Log unique sources
            unique_sources = set(doc.source_file for doc in relevant_docs)
            logger.info(f"Unique source documents: {len(unique_sources)}")
            for source in unique_sources:
                logger.info(f"  - {source}")
            
            logger.info(">>> DECISION: Using DOCUMENT-AWARE RESPONSE")
            
            # Generate response with context
            response = self._generate_document_response(
                user_input, 
                relevant_docs,
                recent_context,
                max_tokens,
                temperature,
                top_p
            )
            
            logger.info(f"Response generated: {len(response)} chars")
            logger.info("="*60)
            return response, relevant_docs
        
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}", exc_info=True)
            return f"I apologize, but I encountered an issue: {str(e)}", []
    
    def _generate_general_response(
        self, 
        user_input: str,
        recent_context: str,
        max_tokens: int,
        temperature: float,
        top_p: float
    ) -> str:
        """Generate response without document context"""
        logger.info("Generating general response (no relevant documents found)")
        prompt = get_general_prompt(user_input, recent_context)
        
        # Log the prompt
        logger.info("="*60)
        logger.info("GENERAL PROMPT SENT TO LLM:")
        logger.info(f"\n{prompt}")
        logger.info("="*60)
        
        response = self.llm_engine.generate(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p
        )
        return utils.clean_response(response)
    
    def _generate_document_response(
        self,
        user_input: str,
        context_docs: List[DocumentChunk],
        recent_context: str,
        max_tokens: int,
        temperature: float,
        top_p: float
    ) -> str:
        """Generate response with document context"""
        logger.info("Generating document-aware response")
        
        # Log the chunks being used
        logger.info("="*60)
        logger.info("CONTEXT CHUNKS USED FOR GENERATION:")
        for i, doc in enumerate(context_docs, 1):
            logger.info(f"\n--- Chunk {i} ---")
            logger.info(f"Source: {doc.source_file}")
            logger.info(f"Chunk ID: {doc.chunk_id}")
            logger.info(f"Relevance Score: {doc.relevance_score:.3f}")
            logger.info(f"Content Length: {len(doc.content)} chars")
            logger.info(f"Content:\n{doc.content}")
        logger.info("="*60)
        
        prompt = get_document_aware_prompt(user_input, context_docs, recent_context)
        
        # Log the full prompt
        logger.info("="*60)
        logger.info("FULL PROMPT SENT TO LLM:")
        logger.info(f"\n{prompt}")
        logger.info("="*60)
        
        response = self.llm_engine.generate(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p
        )
        return utils.clean_response(response)
    
    def add_message(self, role: str, content: str, context_docs: List[DocumentChunk] = None):
        """Add message to conversation history"""
        self.messages.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now(),
            "context_docs": context_docs or []
        })
        
        # Maintain history limit
        if len(self.messages) > MAX_HISTORY * 2:
            self.messages = self.messages[-MAX_HISTORY * 2:]
    
    def get_recent_context(self, num_exchanges: int = 2) -> str:
        """Get recent conversation context"""
        recent_messages = self.messages[-(num_exchanges * 2):]
        context = ""
        for msg in recent_messages:
            role = "Human" if msg["role"] == "user" else "Assistant"
            context += f"{role}: {msg['content']}\n"
        return context
    
    def clear_conversation(self):
        """Clear conversation history"""
        logger.info("Clearing conversation history")
        self.messages.clear()
