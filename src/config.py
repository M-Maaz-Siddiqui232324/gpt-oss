"""Configuration settings for the RAG chatbot"""
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

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

# FastAPI Session Management
SECRET_KEY = os.getenv("SECRET_KEY", "")  
SESSION_MAX_AGE = 300  # 5 minutes (5 * 60 seconds)
MAX_SESSIONS = 10000 
CLEANUP_INTERVAL = 60  # Check every 1 minute (60 seconds)  

# PostgreSQL Database settings (for chatbot sessions)
POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
POSTGRES_PORT = os.getenv("POSTGRES_PORT", "5432")
POSTGRES_DB = os.getenv("POSTGRES_DB", "chatbot")
POSTGRES_SCHEMA = os.getenv("POSTGRES_SCHEMA", "chatbot")
POSTGRES_USER = os.getenv("POSTGRES_USER", "postgres")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "root")
POSTGRES_CONNECTION_STRING = f"host={POSTGRES_HOST} port={POSTGRES_PORT} dbname={POSTGRES_DB} user={POSTGRES_USER} password={POSTGRES_PASSWORD} options='-c search_path={POSTGRES_SCHEMA},public'"

# Default client ID for FlowHCM (set after creating client in database)
DEFAULT_CLIENT_ID = int(os.getenv("DEFAULT_CLIENT_ID", "1"))

# Logging settings
LOG_LEVEL = "INFO"
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
