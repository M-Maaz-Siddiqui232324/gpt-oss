#!/usr/bin/env python3
"""
Startup script for FlowHCM RAG API Server with multiple workers
"""
import subprocess
import sys
import os

def start_server():
    """Start the FastAPI server with uvicorn and multiple workers"""
    print("🚀 Starting FlowHCM RAG API Server with 7 workers...")
    
    # Change to the project directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    # Command to start uvicorn with 7 workers
    cmd = [
        sys.executable, "-m", "uvicorn",
        "apps.api:app",
        "--host", "0.0.0.0",
        "--port", "8000",
        "--workers", "7"
    ]
    
    try:
        # Start the server
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\n⏹️  Server stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error starting server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    start_server()