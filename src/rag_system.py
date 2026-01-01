"""Main RAG system orchestrating all components"""
import logging
import numpy as np
from typing import List, Tuple
from datetime import datetime

from config import (
    DOCS_FOLDER, SEMANTIC_SIMILARITY_THRESHOLD, EMBEDDING_MODEL,
    MODEL_NAME, OLLAMA_BASE_URL, DEFAULT_MAX_TOKENS, DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P, TOP_K_RETRIEVAL, TOP_K_CONTEXT, MIN_RELEVANCE_THRESHOLD,
    BEST_MATCH_THRESHOLD
)
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
    """Complete RAG system with per-client embedding support"""
    
    def __init__(self):
        self.doc_processor = DocumentProcessor(DOCS_FOLDER)
        self.chunker = SemanticChunker(similarity_threshold=SEMANTIC_SIMILARITY_THRESHOLD)
        self.vector_store = VectorStore(EMBEDDING_MODEL)
        self.llm_engine = LLMEngine(MODEL_NAME, OLLAMA_BASE_URL)
        self.retriever = SemanticRetriever(self.vector_store)
        self.current_client = None
        
        self.documents = []
        self.chunks = []
        
        # Load LLM immediately
        if not self.llm_engine.load_model():
            logger.error("Failed to load LLM model")
        
        # Initialize general index from docs folder
        self._initialize_general_index()
    
    def _initialize_general_index(self):
        """Initialize general index from docs folder (shared across all clients)"""
        # Try to load existing general index
        if not self.vector_store._load_general_index():
            # No general index exists, create one from docs folder
            folder_documents = self.doc_processor.load_documents()
            
            if folder_documents:
                folder_chunks = self.chunker.create_chunks(folder_documents)
                if folder_chunks:
                    self.vector_store.build_general_index(folder_chunks, force_rebuild=True)
    
    def load_client_index(self, company_pin: str) -> bool:
        """
        Load a specific client's HR policy index
        
        Args:
            company_pin: Client's company PIN
            
        Returns:
            True if loaded successfully, False otherwise
        """
        # Set client-specific paths
        self.vector_store.set_client_paths(company_pin)
        self.current_client = company_pin
        
        # Try to load existing client index (HR policies)
        return self.vector_store._load_client_index()
    

    

    def query_with_context(
        self, 
        user_input: str,
        recent_context: str = "",
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
        top_p: float = DEFAULT_TOP_P
    ) -> Tuple[str, List[DocumentChunk]]:
        """Process a user query with external context (for session management)"""
        try:
            # Retrieve relevant documents
            context_docs = self.retriever.retrieve(user_input, TOP_K_RETRIEVAL)
            
            if not context_docs:
                logger.info(f"🎯 Prompt Selection: GENERAL (no documents found)")
                return self._generate_general_response(user_input, recent_context, max_tokens, temperature, top_p), []
            
            # Check if best match is good enough (best score check)
            best_score = max(doc.relevance_score for doc in context_docs)
            if best_score < BEST_MATCH_THRESHOLD:
                logger.info(f"🎯 Prompt Selection: GENERAL (best score {best_score:.3f} below threshold {BEST_MATCH_THRESHOLD})")
                return self._generate_general_response(user_input, recent_context, max_tokens, temperature, top_p), []
            
            scores = [doc.relevance_score for doc in context_docs]
            mean_score = np.mean(scores)
            std_score = np.std(scores)
            
            dynamic_threshold = max(MIN_RELEVANCE_THRESHOLD, mean_score - 0.5 * std_score)
            
            logger.info(f"📊 Relevance Threshold Analysis:")
            logger.info(f"  Retrieved Documents: {len(context_docs)}")
            logger.info(f"  Score Range: {min(scores):.3f} - {max(scores):.3f}")
            logger.info(f"  Mean Score: {mean_score:.3f}")
            logger.info(f"  Std Deviation: {std_score:.3f}")
            logger.info(f"  Dynamic Threshold: {dynamic_threshold:.3f}")
            logger.info(f"  Min Threshold: {MIN_RELEVANCE_THRESHOLD}")
            
            relevant_docs = [doc for doc in context_docs if doc.relevance_score >= dynamic_threshold]
            relevant_docs = relevant_docs[:TOP_K_CONTEXT]
            
            if not relevant_docs:
                logger.info(f"🎯 Prompt Selection: GENERAL (no documents above threshold {dynamic_threshold:.3f})")
                return self._generate_general_response(user_input, recent_context, max_tokens, temperature, top_p), []
            
            logger.info(f"🎯 Prompt Selection: DOCUMENT-AWARE")
            logger.info(f"  Documents Above Threshold: {len(relevant_docs)}")
            logger.info(f"  Selected Documents: {[doc.source_file for doc in relevant_docs]}")
            
            # Generate response with context
            response = self._generate_document_response(
                user_input, 
                relevant_docs,
                recent_context,
                max_tokens,
                temperature,
                top_p
            )
            
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
        prompt = get_general_prompt(user_input, recent_context)
        
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
        prompt = get_document_aware_prompt(user_input, context_docs, recent_context)
        
        response = self.llm_engine.generate(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p
        )
        return utils.clean_response(response)

