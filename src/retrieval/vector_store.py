"""FAISS vector store for semantic search"""
import os
import pickle
import logging
import numpy as np
from typing import List, Optional

from config import get_client_index_path, get_client_chunks_path, get_general_index_path, get_general_chunks_path

logger = logging.getLogger(__name__)

try:
    import faiss
    from sentence_transformers import SentenceTransformer
    VECTOR_SEARCH_AVAILABLE = True
except ImportError:
    VECTOR_SEARCH_AVAILABLE = False
    logger.warning("FAISS or sentence-transformers not available")


class VectorStore:
    """Manages dual FAISS indexes for semantic search (general + client-specific)"""
    
    def __init__(self, embedding_model: str):
        self.embedding_model_name = embedding_model
        self.encoder = None
        
        # General index (docs folder - shared across all clients)
        self.general_index = None
        self.general_chunks = []
        self.general_index_file = get_general_index_path()
        self.general_chunks_file = get_general_chunks_path()
        
        # Client-specific index (HR policies from API)
        self.client_index = None
        self.client_chunks = []
        self.client_index_file = None
        self.client_chunks_file = None
        self.current_client = None
        
        if VECTOR_SEARCH_AVAILABLE:
            self.encoder = SentenceTransformer(embedding_model)
        else:
            logger.error("Vector search dependencies not available")
    
    def set_client_paths(self, company_pin: str):
        """
        Set paths for a specific client's HR policy embeddings
        
        Args:
            company_pin: Client's company PIN
        """
        self.client_index_file = get_client_index_path(company_pin)
        self.client_chunks_file = get_client_chunks_path(company_pin)
        self.current_client = company_pin
    
    def build_general_index(self, chunks: List, force_rebuild: bool = False) -> bool:
        """Build general FAISS index from docs folder chunks"""
        if not VECTOR_SEARCH_AVAILABLE:
            logger.error("Cannot build index - dependencies missing")
            return False
        
        # Load existing general index (unless force_rebuild is True)
        if not force_rebuild and self._load_general_index():
            return True
        
        if not chunks:
            return False
        
        self.general_chunks = chunks
        
        # Create embeddings
        texts = [chunk.content for chunk in chunks]
        embeddings = self.encoder.encode(
            texts, 
            show_progress_bar=False, 
            convert_to_numpy=True
        )
        
        # Build FAISS index
        dimension = embeddings.shape[1]
        self.general_index = faiss.IndexFlatIP(dimension)
        
        faiss.normalize_L2(embeddings)
        self.general_index.add(embeddings.astype('float32'))
        
        self._save_general_index()
        return True
    
    def build_client_index(self, chunks: List, force_rebuild: bool = True) -> bool:
        """Build client-specific FAISS index from HR policy chunks"""
        if not VECTOR_SEARCH_AVAILABLE:
            logger.error("Cannot build index - dependencies missing")
            return False
        
        if not chunks:
            return False
        
        self.client_chunks = chunks
        
        # Create embeddings
        texts = [chunk.content for chunk in chunks]
        embeddings = self.encoder.encode(
            texts, 
            show_progress_bar=False, 
            convert_to_numpy=True
        )
        
        # Build FAISS index
        dimension = embeddings.shape[1]
        self.client_index = faiss.IndexFlatIP(dimension)
        
        faiss.normalize_L2(embeddings)
        self.client_index.add(embeddings.astype('float32'))
        
        self._save_client_index()
        return True
    
    def search(self, query: str, top_k: int = 12) -> List[tuple]:
        """Search both general and client-specific indexes and combine results"""
        if not VECTOR_SEARCH_AVAILABLE:
            logger.warning("Vector search not available")
            return []
        
        try:
            query_embedding = self.encoder.encode([query], convert_to_numpy=True)
            faiss.normalize_L2(query_embedding)
            
            all_results = []
            
            # Search general index (docs folder)
            if self.general_index is not None:
                scores, indices = self.general_index.search(query_embedding.astype('float32'), top_k)
                for score, idx in zip(scores[0], indices[0]):
                    if idx < len(self.general_chunks) and score > 0:
                        all_results.append(('general', idx, float(score)))
            
            # Search client-specific index (HR policies)
            if self.client_index is not None:
                scores, indices = self.client_index.search(query_embedding.astype('float32'), top_k)
                for score, idx in zip(scores[0], indices[0]):
                    if idx < len(self.client_chunks) and score > 0:
                        all_results.append(('client', idx, float(score)))
            
            # Sort by score and return top results
            all_results.sort(key=lambda x: x[2], reverse=True)
            return all_results[:top_k]
        
        except Exception as e:
            logger.error(f"Semantic search error: {e}")
            return []
    
    def _save_general_index(self):
        """Save general FAISS index and chunks to disk"""
        try:
            # Create directory for index file
            index_dir = os.path.dirname(self.general_index_file)
            if index_dir and not os.path.exists(index_dir):
                os.makedirs(index_dir, exist_ok=True)
            
            if self.general_index is not None:
                faiss.write_index(self.general_index, self.general_index_file)
            
            with open(self.general_chunks_file, 'wb') as f:
                pickle.dump(self.general_chunks, f)
        
        except Exception as e:
            logger.error(f"Failed to save general index: {str(e)}", exc_info=True)
    
    def _save_client_index(self):
        """Save client-specific FAISS index and chunks to disk"""
        try:
            # Create directory for index file
            index_dir = os.path.dirname(self.client_index_file)
            if index_dir and not os.path.exists(index_dir):
                os.makedirs(index_dir, exist_ok=True)
            
            if self.client_index is not None:
                faiss.write_index(self.client_index, self.client_index_file)
            
            with open(self.client_chunks_file, 'wb') as f:
                pickle.dump(self.client_chunks, f)
        
        except Exception as e:
            logger.error(f"Failed to save client index: {str(e)}", exc_info=True)
    
    def _load_general_index(self) -> bool:
        """Load general FAISS index and chunks from disk"""
        try:
            if os.path.exists(self.general_index_file) and os.path.exists(self.general_chunks_file):
                self.general_index = faiss.read_index(self.general_index_file)
                
                with open(self.general_chunks_file, 'rb') as f:
                    self.general_chunks = pickle.load(f)
                
                return True
        
        except Exception as e:
            logger.warning(f"Failed to load general index: {str(e)}")
        
        return False
    
    def _load_client_index(self) -> bool:
        """Load client-specific FAISS index and chunks from disk"""
        try:
            if (self.client_index_file and self.client_chunks_file and 
                os.path.exists(self.client_index_file) and os.path.exists(self.client_chunks_file)):
                
                self.client_index = faiss.read_index(self.client_index_file)
                
                with open(self.client_chunks_file, 'rb') as f:
                    self.client_chunks = pickle.load(f)
                
                return True
        
        except Exception as e:
            logger.warning(f"Failed to load client index: {str(e)}")
        
        return False
    
    def rebuild_client_index_from_scratch(self, hr_policy_chunks: List) -> bool:
        """
        Rebuild client-specific FAISS index from scratch (used for FlowHCM sync)
        Deletes existing client index and creates fresh one with HR policy chunks
        
        Args:
            hr_policy_chunks: Complete list of HR policy chunks to index
            
        Returns:
            True if successful, False otherwise
        """
        if not VECTOR_SEARCH_AVAILABLE:
            logger.error("Cannot rebuild index - dependencies missing")
            return False
        
        if not hr_policy_chunks:
            logger.warning("No HR policy chunks provided for rebuild")
            return False
        
        try:
            # Clear existing client index and chunks
            self.client_index = None
            self.client_chunks = []
            
            # Delete existing index files from disk
            if self.client_index_file and os.path.exists(self.client_index_file):
                os.remove(self.client_index_file)
                logger.info(f"Deleted existing client index file: {self.client_index_file}")

            if self.client_chunks_file and os.path.exists(self.client_chunks_file):
                os.remove(self.client_chunks_file)
                logger.info(f"Deleted existing client chunks file: {self.client_chunks_file}")
            
            # Build fresh client index
            return self.build_client_index(hr_policy_chunks, force_rebuild=True)
        
        except Exception as e:
            logger.error(f"Failed to rebuild client index: {e}", exc_info=True)
            return False
    

