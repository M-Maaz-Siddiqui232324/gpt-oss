"""FAISS vector store for semantic search"""
import os
import pickle
import logging
import numpy as np
from typing import List, Optional

logger = logging.getLogger(__name__)

try:
    import faiss
    from sentence_transformers import SentenceTransformer
    VECTOR_SEARCH_AVAILABLE = True
except ImportError:
    VECTOR_SEARCH_AVAILABLE = False
    logger.warning("FAISS or sentence-transformers not available")


class VectorStore:
    """Manages FAISS index for semantic search"""
    
    def __init__(self, embedding_model: str, index_file: str, chunks_file: str):
        self.embedding_model_name = embedding_model
        self.index_file = index_file
        self.chunks_file = chunks_file
        self.encoder = None
        self.index = None
        self.chunks = []
        
        if VECTOR_SEARCH_AVAILABLE:
            logger.info(f"Loading embedding model: {embedding_model}")
            self.encoder = SentenceTransformer(embedding_model)
            logger.info("Embedding model loaded successfully")
        else:
            logger.error("Vector search dependencies not available")
    
    def build_index(self, chunks: List) -> bool:
        """Build FAISS index from chunks"""
        if not VECTOR_SEARCH_AVAILABLE:
            logger.error("Cannot build index - dependencies missing")
            return False
        
        # Try loading existing index
        if self._load_index():
            logger.info("Loaded existing FAISS index from disk")
            return True
        
        if not chunks:
            logger.warning("No chunks available to build index")
            return False
        
        logger.info(f"Creating embeddings for {len(chunks)} chunks")
        self.chunks = chunks
        
        # Create embeddings
        texts = [chunk.content for chunk in chunks]
        embeddings = self.encoder.encode(
            texts, 
            show_progress_bar=True, 
            convert_to_numpy=True
        )
        logger.info(f"Embeddings created with shape: {embeddings.shape}")
        
        # Build FAISS index
        dimension = embeddings.shape[1]
        logger.info(f"Building FAISS index with dimension: {dimension}")
        self.index = faiss.IndexFlatIP(dimension)
        
        # Normalize for cosine similarity
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings.astype('float32'))
        logger.info(f"FAISS index built with {self.index.ntotal} vectors")
        
        # Save index
        self._save_index()
        return True
    
    def search(self, query: str, top_k: int = 12) -> List[tuple]:
        """Search for similar chunks"""
        if not VECTOR_SEARCH_AVAILABLE or self.index is None:
            logger.warning("Vector search not available")
            return []
        
        try:
            logger.debug(f"Semantic search for: '{query}' (top_k={top_k})")
            
            # Encode query
            query_embedding = self.encoder.encode([query], convert_to_numpy=True)
            faiss.normalize_L2(query_embedding)
            
            # Search
            scores, indices = self.index.search(query_embedding.astype('float32'), top_k * 2)
            
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < len(self.chunks) and score > 0:
                    results.append((idx, float(score)))
            
            logger.info(f"Semantic search found {len(results)} candidates")
            return results
        
        except Exception as e:
            logger.error(f"Semantic search error: {e}")
            return []
    
    def _save_index(self):
        """Save FAISS index and chunks to disk"""
        try:
            # Create data directory if it doesn't exist
            index_dir = os.path.dirname(self.index_file)
            if index_dir and not os.path.exists(index_dir):
                os.makedirs(index_dir)
                logger.info(f"Created directory: {index_dir}")
            
            if self.index is not None:
                faiss.write_index(self.index, self.index_file)
                logger.info(f"Saved FAISS index to: {self.index_file}")
            
            with open(self.chunks_file, 'wb') as f:
                pickle.dump(self.chunks, f)
            logger.info(f"Saved chunks to: {self.chunks_file}")
        
        except Exception as e:
            logger.error(f"Failed to save index: {str(e)}")
    
    def _load_index(self) -> bool:
        """Load FAISS index and chunks from disk"""
        try:
            if os.path.exists(self.index_file) and os.path.exists(self.chunks_file):
                logger.info("Loading existing FAISS index and chunks")
                self.index = faiss.read_index(self.index_file)
                
                with open(self.chunks_file, 'rb') as f:
                    self.chunks = pickle.load(f)
                
                logger.info(f"Loaded {self.index.ntotal} vectors and {len(self.chunks)} chunks")
                return True
        
        except Exception as e:
            logger.warning(f"Failed to load existing index: {str(e)}")
        
        return False
