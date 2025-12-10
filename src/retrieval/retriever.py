"""Semantic retrieval using vector search"""
import logging
from typing import List

logger = logging.getLogger(__name__)


class SemanticRetriever:
    """Semantic search using vector similarity"""
    
    def __init__(self, vector_store, chunks: List):
        self.vector_store = vector_store
        self.chunks = chunks

    
    def retrieve(self, query: str, top_k: int = 12) -> List:
        """Semantic search using vector similarity"""
        semantic_results = self.vector_store.search(query, top_k)
        
        if not semantic_results:
            return []
        
        results = []
        for idx, score in semantic_results:
            if idx < len(self.chunks):
                chunk = self.chunks[idx]
                chunk.relevance_score = score
                results.append(chunk)
        
        return results
