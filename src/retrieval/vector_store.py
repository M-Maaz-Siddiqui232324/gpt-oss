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
    
    def __init__(self, embedding_model: str, index_file: str = None, chunks_file: str = None):
        self.embedding_model_name = embedding_model
        self.index_file = index_file
        self.chunks_file = chunks_file
        self.encoder = None
        self.index = None
        self.chunks = []
        self.current_client = None  # Track which client's index is loaded
        
        if VECTOR_SEARCH_AVAILABLE:
            logger.info(f"Loading embedding model: {embedding_model}")
            self.encoder = SentenceTransformer(embedding_model)
            logger.info("Embedding model loaded successfully")
        else:
            logger.error("Vector search dependencies not available")
    
    def set_client_paths(self, company_pin: str):
        """
        Set paths for a specific client's embeddings
        
        Args:
            company_pin: Client's company PIN
        """
        from config import get_client_index_path, get_client_chunks_path
        self.index_file = get_client_index_path(company_pin)
        self.chunks_file = get_client_chunks_path(company_pin)
        self.current_client = company_pin
        logger.info(f"Set client paths for: {company_pin}")
        logger.info(f"  Index: {self.index_file}")
        logger.info(f"  Chunks: {self.chunks_file}")
    
    def build_index(self, chunks: List, force_rebuild: bool = False) -> bool:
        """Build FAISS index from chunks"""
        if not VECTOR_SEARCH_AVAILABLE:
            logger.error("Cannot build index - dependencies missing")
            return False
        
        # load existing index (unless force_rebuild is True)
        if not force_rebuild and self._load_index():
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
        
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings.astype('float32'))
        logger.info(f"FAISS index built with {self.index.ntotal} vectors")
        
        self._save_index()
        return True
    
    def search(self, query: str, top_k: int = 12) -> List[tuple]:
        """Search for similar chunks"""
        if not VECTOR_SEARCH_AVAILABLE or self.index is None:
            logger.warning("Vector search not available")
            return []
        
        try:
            logger.debug(f"Semantic search for: '{query}' (top_k={top_k})")
            
            query_embedding = self.encoder.encode([query], convert_to_numpy=True)
            faiss.normalize_L2(query_embedding)
            
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
            # Create directory for index file
            index_dir = os.path.dirname(self.index_file)
            if index_dir and not os.path.exists(index_dir):
                os.makedirs(index_dir, exist_ok=True)
                logger.info(f"Created directory: {index_dir}")
            
            # Create directory for chunks file (might be different)
            chunks_dir = os.path.dirname(self.chunks_file)
            if chunks_dir and not os.path.exists(chunks_dir):
                os.makedirs(chunks_dir, exist_ok=True)
                logger.info(f"Created directory: {chunks_dir}")
            
            if self.index is not None:
                logger.info(f"Saving FAISS index to: {self.index_file}")
                faiss.write_index(self.index, self.index_file)
                logger.info(f"✅ Saved FAISS index to: {self.index_file}")
            
            logger.info(f"Saving chunks to: {self.chunks_file}")
            with open(self.chunks_file, 'wb') as f:
                pickle.dump(self.chunks, f)
            logger.info(f"✅ Saved chunks to: {self.chunks_file}")
        
        except Exception as e:
            logger.error(f"❌ Failed to save index: {str(e)}", exc_info=True)
    
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
    
    def rebuild_index_from_scratch(self, all_chunks: List) -> bool:
        """
        Rebuild FAISS index from scratch (used for FlowHCM sync)
        Deletes existing index and creates fresh one with all chunks
        
        Args:
            all_chunks: Complete list of all chunks to index
            
        Returns:
            True if successful, False otherwise
        """
        if not VECTOR_SEARCH_AVAILABLE:
            logger.error("❌ Cannot rebuild index - dependencies missing")
            return False
        
        if not all_chunks:
            logger.warning("⚠️  No chunks provided for rebuild")
            return False
        
        try:
            logger.info("="*60)
            logger.info("🔄 REBUILDING INDEX FROM SCRATCH")
            logger.info("="*60)
            logger.info(f"Total chunks to index: {len(all_chunks)}")
            
            # Clear existing index and chunks
            logger.info("🗑️  Clearing existing index and chunks...")
            self.index = None
            self.chunks = []
            logger.info("✅ Cleared existing data")
            
            # Build fresh index (force rebuild, don't load from disk)
            logger.info("🔨 Building fresh FAISS index...")
            return self.build_index(all_chunks, force_rebuild=True)
        
        except Exception as e:
            logger.error(f"❌ Failed to rebuild index: {e}", exc_info=True)
            return False
    
    def add_documents(self, new_chunks: List) -> bool:
        """
        Add new document chunks to existing FAISS index (incremental indexing)
        
        Args:
            new_chunks: List of new chunks to add
            
        Returns:
            True if successful, False otherwise
        """
        if not VECTOR_SEARCH_AVAILABLE:
            logger.error("❌ Cannot add documents - dependencies missing")
            return False
        
        if not new_chunks:
            logger.warning("⚠️  No new chunks to add")
            return False
        
        try:
            logger.info(f"📊 Starting incremental indexing for {len(new_chunks)} new chunks")
            
            # Load existing index if not already loaded
            if self.index is None:
                logger.info("🔍 Index not loaded in memory, attempting to load from disk...")
                if not self._load_index():
                    logger.warning("⚠️  No existing index found, creating new one from scratch")
                    return self.build_index(new_chunks)
                logger.info(f"✅ Loaded existing index with {self.index.ntotal} vectors")
            else:
                logger.info(f"✅ Using already loaded index with {self.index.ntotal} vectors")
            
            # Store original counts for comparison
            original_vector_count = self.index.ntotal
            original_chunk_count = len(self.chunks)
            
            # Generate embeddings for new chunks
            logger.info("🔢 Generating embeddings for new chunks...")
            logger.info(f"   Embedding model: {self.embedding_model_name}")
            texts = [chunk.content for chunk in new_chunks]
            logger.info(f"   Processing {len(texts)} text chunks")
            
            new_embeddings = self.encoder.encode(
                texts,
                show_progress_bar=True,
                convert_to_numpy=True
            )
            logger.info(f"✅ Generated embeddings with shape: {new_embeddings.shape}")
            logger.info(f"   Embedding dimension: {new_embeddings.shape[1]}")
            logger.info(f"   Data type: {new_embeddings.dtype}")
            
            # Normalize and add to index
            logger.info("🔄 Normalizing embeddings (L2 normalization)...")
            faiss.normalize_L2(new_embeddings)
            logger.info("✅ Embeddings normalized")
            
            logger.info(f"➕ Adding {len(new_chunks)} vectors to FAISS index...")
            self.index.add(new_embeddings.astype('float32'))
            new_vector_count = self.index.ntotal
            logger.info(f"✅ Vectors added successfully")
            logger.info(f"   Previous vector count: {original_vector_count}")
            logger.info(f"   New vector count: {new_vector_count}")
            logger.info(f"   Vectors added: {new_vector_count - original_vector_count}")
            
            # Append chunks to existing chunks list
            logger.info("📝 Appending chunks to chunk list...")
            self.chunks.extend(new_chunks)
            new_chunk_count = len(self.chunks)
            logger.info(f"✅ Chunks appended")
            logger.info(f"   Previous chunk count: {original_chunk_count}")
            logger.info(f"   New chunk count: {new_chunk_count}")
            logger.info(f"   Chunks added: {new_chunk_count - original_chunk_count}")
            
            # Log chunk details
            logger.info("📋 New chunk details:")
            for i, chunk in enumerate(new_chunks[:3], 1):  # Log first 3 chunks
                logger.info(f"   Chunk {i}:")
                logger.info(f"     Source: {chunk.source_file}")
                logger.info(f"     Chunk ID: {chunk.chunk_id}")
                logger.info(f"     Content length: {len(chunk.content)} chars")
                logger.info(f"     Content preview: {chunk.content[:100]}...")
            if len(new_chunks) > 3:
                logger.info(f"   ... and {len(new_chunks) - 3} more chunks")
            
            # Save updated index and chunks
            logger.info("💾 Saving updated index and chunks to disk...")
            self._save_index()
            logger.info("✅ Successfully saved updated index")
            logger.info(f"   Index file: {self.index_file}")
            logger.info(f"   Chunks file: {self.chunks_file}")
            
            logger.info("="*60)
            logger.info("✅ INCREMENTAL INDEXING COMPLETED SUCCESSFULLY")
            logger.info(f"   Total vectors in index: {self.index.ntotal}")
            logger.info(f"   Total chunks: {len(self.chunks)}")
            logger.info("="*60)
            
            return True
        
        except Exception as e:
            logger.error(f"❌ Failed to add documents to index: {e}", exc_info=True)
            logger.error(f"   Error type: {type(e).__name__}")
            logger.error(f"   Error details: {str(e)}")
            return False
