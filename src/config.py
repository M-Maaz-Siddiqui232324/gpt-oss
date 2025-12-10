"""Configuration settings for the RAG chatbot"""
import os
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Model settings
MODEL_NAME = "gpt-oss:20b"  
OLLAMA_BASE_URL = "http://localhost:11434" 
EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"

DOCS_FOLDER = os.path.join(PROJECT_ROOT, "docs")
SEMANTIC_SIMILARITY_THRESHOLD = 0.7  # Lower = more chunks, Higher = fewer chunks

# Per-client embedding storage
DATA_FOLDER = os.path.join(PROJECT_ROOT, "data")

# Legacy paths (kept for backward compatibility)
FAISS_INDEX_FILE = os.path.join(PROJECT_ROOT, "data", "faiss_index.bin")
CHUNKS_FILE = os.path.join(PROJECT_ROOT, "data", "document_chunks.pkl")

def get_client_index_path(company_pin: str) -> str:
    """Get FAISS index path for a specific client (HR policies only)"""
    return os.path.join(DATA_FOLDER, company_pin, "hr_policies_index.bin")

def get_client_chunks_path(company_pin: str) -> str:
    """Get chunks file path for a specific client (HR policies only)"""
    return os.path.join(DATA_FOLDER, company_pin, "hr_policies_chunks.pkl")

def get_general_index_path() -> str:
    """Get FAISS index path for general docs (shared across all clients)"""
    return os.path.join(DATA_FOLDER, "general", "docs_index.bin")

def get_general_chunks_path() -> str:
    """Get chunks file path for general docs (shared across all clients)"""
    return os.path.join(DATA_FOLDER, "general", "docs_chunks.pkl")

DEFAULT_MAX_TOKENS = 750
DEFAULT_TEMPERATURE = 0.1
DEFAULT_TOP_P = 0.7
DEFAULT_TOP_K = 20
REPETITION_PENALTY = 1.2
NO_REPEAT_NGRAM_SIZE = 3
OLLAMA_TIMEOUT = 500

TOP_K_RETRIEVAL = 10
TOP_K_CONTEXT = 7  
MIN_RELEVANCE_THRESHOLD = 0.6

MAX_HISTORY = 5
RECENT_CONTEXT_EXCHANGES = 5

API_HOST = "0.0.0.0"
API_PORT = 8000

# FastAPI Session Management
SECRET_KEY = os.getenv("SECRET_KEY", "")  

SESSION_MAX_AGE = 1000000 
MAX_SESSIONS = 10000 
CLEANUP_INTERVAL = 60  

# PostgreSQL Database settings (for chatbot sessions)
POSTGRES_HOST = os.getenv("POSTGRES_HOST")
POSTGRES_PORT = os.getenv("POSTGRES_PORT")
POSTGRES_DB = os.getenv("POSTGRES_DB")
POSTGRES_SCHEMA = os.getenv("POSTGRES_SCHEMA")
POSTGRES_USER = os.getenv("POSTGRES_USER")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD")

# Validate required environment variables
if not all([POSTGRES_HOST, POSTGRES_PORT, POSTGRES_DB, POSTGRES_SCHEMA, POSTGRES_USER, POSTGRES_PASSWORD]):
    raise ValueError("Missing required PostgreSQL environment variables. Check your .env file.")

POSTGRES_CONNECTION_STRING = f"host={POSTGRES_HOST} port={POSTGRES_PORT} dbname={POSTGRES_DB} user={POSTGRES_USER} password={POSTGRES_PASSWORD} options='-c search_path={POSTGRES_SCHEMA},public'"

LOG_LEVEL = "INFO"
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
