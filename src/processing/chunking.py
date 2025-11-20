"""Pure semantic chunking using sentence embeddings and similarity"""
import logging
import re
from typing import List
from dataclasses import dataclass
import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


@dataclass
class DocumentChunk:
    """Document chunk with metadata"""
    content: str
    source_file: str
    chunk_id: int
    file_type: str
    relevance_score: float = 0.0


class SemanticChunker:
    """Creates semantic chunks by analyzing sentence similarity"""
    
    def __init__(self, chunk_size: int = None, overlap: int = None, similarity_threshold: float = 0.7):
        self.similarity_threshold = similarity_threshold
        logger.info(f"Initialized SemanticChunker (pure semantic mode, threshold={similarity_threshold})")
        logger.info("Loading sentence embedding model for semantic analysis...")
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')  # Fast, lightweight model
        logger.info("Sentence embedding model loaded")
    
    def create_chunks(self, documents: List[dict]) -> List[DocumentChunk]:
        """Create semantic chunks from documents"""
        logger.info(f"Creating semantic chunks from {len(documents)} documents")
        chunks = []
        
        for doc in documents:
            doc_chunks = self._chunk_document(doc)
            chunks.extend(doc_chunks)
            logger.debug(f"Created {len(doc_chunks)} chunks from {doc['name']}")
        
        logger.info(f"Total chunks created: {len(chunks)}")
        return chunks
    
    def _chunk_document(self, doc: dict) -> List[DocumentChunk]:
        """Chunk document using semantic similarity between sentences"""
        # Split into sentences
        sentences = self._split_into_sentences(doc['content'])
        
        if len(sentences) <= 1:
            # Single sentence or empty - return as one chunk
            return [DocumentChunk(
                content=doc['content'].strip(),
                source_file=doc['name'],
                chunk_id=0,
                file_type=doc['type']
            )]
        
        logger.debug(f"Analyzing {len(sentences)} sentences for semantic boundaries...")
        
        # Encode all sentences
        embeddings = self.encoder.encode(sentences, convert_to_numpy=True)
        
        # Find semantic boundaries by comparing consecutive sentences
        chunks = []
        current_chunk = [sentences[0]]
        chunk_id = 0
        
        for i in range(1, len(sentences)):
            # Calculate cosine similarity between current and previous sentence
            similarity = self._cosine_similarity(embeddings[i-1], embeddings[i])
            
            if similarity < self.similarity_threshold:
                # Low similarity = semantic boundary detected
                # Save current chunk and start new one
                chunk_content = ' '.join(current_chunk).strip()
                if chunk_content:
                    chunks.append(DocumentChunk(
                        content=chunk_content,
                        source_file=doc['name'],
                        chunk_id=chunk_id,
                        file_type=doc['type']
                    ))
                    chunk_id += 1
                
                current_chunk = [sentences[i]]
                logger.debug(f"Semantic boundary detected at sentence {i} (similarity: {similarity:.3f})")
            else:
                # High similarity = continue current chunk
                current_chunk.append(sentences[i])
        
        # Add final chunk
        if current_chunk:
            chunk_content = ' '.join(current_chunk).strip()
            if chunk_content:
                chunks.append(DocumentChunk(
                    content=chunk_content,
                    source_file=doc['name'],
                    chunk_id=chunk_id,
                    file_type=doc['type']
                ))
        
        logger.debug(f"Document '{doc['name']}' split into {len(chunks)} semantic chunks")
        return chunks
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        # Simple sentence splitting (can be improved with spaCy/NLTK)
        # Split on . ! ? followed by space and capital letter
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors"""
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
