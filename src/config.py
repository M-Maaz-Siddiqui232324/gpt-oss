"""Configuration settings for the RAG chatbot"""
import os

# Get project root directory (parent of src folder)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Model settings
MODEL_NAME = "gpt-oss:20b"  # Ollama model name - local version
OLLAMA_BASE_URL = "http://localhost:11434"  # Ollama API endpoint
EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"

# Document settings
DOCS_FOLDER = os.path.join(PROJECT_ROOT, "docs")
SEMANTIC_SIMILARITY_THRESHOLD = 0.7  # Lower = more chunks, Higher = fewer chunks

# Vector store settings
FAISS_INDEX_FILE = os.path.join(PROJECT_ROOT, "data", "faiss_index.bin")
CHUNKS_FILE = os.path.join(PROJECT_ROOT, "data", "document_chunks.pkl")

# Generation settings
DEFAULT_MAX_TOKENS = 750
DEFAULT_TEMPERATURE = 0.2
DEFAULT_TOP_P = 0.7
DEFAULT_TOP_K = 20
REPETITION_PENALTY = 1.2
NO_REPEAT_NGRAM_SIZE = 3
OLLAMA_TIMEOUT = 120  # Timeout in seconds for Ollama API requests (max time to wait for response)

# RAG settings
TOP_K_RETRIEVAL = 10
TOP_K_CONTEXT = 7  # Use top 7 most relevant chunks for context
MIN_RELEVANCE_THRESHOLD = 0.6

# Conversation settings
MAX_HISTORY = 5
RECENT_CONTEXT_EXCHANGES = 5

# API settings
API_HOST = "0.0.0.0"
API_PORT = 8000

# Redis settings (Session Management)
REDIS_HOST = "localhost"
REDIS_PORT = 6379
REDIS_DB = 0
SESSION_EXPIRY_SECONDS = 1800  # 30 minutes (fallback only)

# Session Archive settings
ARCHIVE_FOLDER = os.path.join(PROJECT_ROOT, "session_archives")
ARCHIVE_FORMAT = "json"  # json or txt

# Logging settings
LOG_LEVEL = "INFO"
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
