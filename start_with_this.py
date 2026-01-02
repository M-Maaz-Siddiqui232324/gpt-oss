#!/usr/bin/env python3
"""
Production startup script with uvicorn
"""
import os
import sys
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from config import API_HOST, API_PORT

logger = logging.getLogger(__name__)

def start_with_uvicorn():
    """Start with uvicorn"""
    try:
        import uvicorn
        
        worker_count = 4
        logger.info(f"🚀 Starting uvicorn server on {API_HOST}:{API_PORT}")
        logger.info(f"👥 Workers: {worker_count}")
        logger.info(f"🔧 Process ID: {os.getpid()}")
        
        uvicorn.run(
            "apps.api:app", 
            host=API_HOST,
            port=API_PORT,
            workers=worker_count,
            reload=False,
            log_level="info",
            access_log=True
        )
        
    except Exception as e:
        logger.error(f"Failed to start server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    print("🚀 Starting FlowHCM RAG Chatbot API (Production Mode)")
    print(f"📍 Server: {API_HOST}:{API_PORT}")
    print("👥 Workers: 4")
    print("🔧 Mode: Production")
    print("-" * 50)
    
    start_with_uvicorn()