"""Semantic retrieval using vector search"""
import logging
from typing import List

logger = logging.getLogger(__name__)


class SemanticRetriever:
    """Semantic search using vector similarity"""
    
    def __init__(self, vector_store, chunks: List):
        self.vector_store = vector_store
        self.chunks = chunks
        logger.info("Initialized SemanticRetriever")
    
    def retrieve(self, query: str, top_k: int = 12) -> List:
        """Semantic search using vector similarity"""
        logger.info(f"Semantic search for: '{query}' (top_k={top_k})")
        
        # Get semantic scores from vector store
        semantic_results = self.vector_store.search(query, top_k)
        
        if not semantic_results:
            logger.warning("No semantic results found")
            return []
        
        logger.info(f"Found {len(semantic_results)} semantic matches")
        
        # Build results with scores
        results = []
        for idx, score in semantic_results:
            if idx < len(self.chunks):
                chunk = self.chunks[idx]
                chunk.relevance_score = score
                results.append(chunk)
                logger.debug(f"Result {len(results)}: score={score:.3f}, source={chunk.source_file}")
        
        logger.info(f"Returning {len(results)} results")
        return results
