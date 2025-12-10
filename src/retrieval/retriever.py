"""Semantic retrieval using vector search"""
import logging
from typing import List

logger = logging.getLogger(__name__)


class SemanticRetriever:
    """Semantic search using vector similarity"""
    
    def __init__(self, vector_store):
        self.vector_store = vector_store

    
    def retrieve(self, query: str, top_k: int = 12) -> List:
        """Semantic search using vector similarity across both general and client indexes"""
        semantic_results = self.vector_store.search(query, top_k)
        
        if not semantic_results:
            return []
        
        results = []
        for source_type, idx, score in semantic_results:
            if source_type == 'general' and idx < len(self.vector_store.general_chunks):
                chunk = self.vector_store.general_chunks[idx]
                chunk.relevance_score = score
                results.append(chunk)
            elif source_type == 'client' and idx < len(self.vector_store.client_chunks):
                chunk = self.vector_store.client_chunks[idx]
                chunk.relevance_score = score
                results.append(chunk)
        
        return results
