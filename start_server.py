#!/usr/bin/env python3
"""
Optimized startup script for FlowHCM RAG API Server with maximum performance
"""
import subprocess
import sys
import os
import time
import requests

def check_ollama_status():
    """Check if Ollama is running"""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        return response.status_code == 200
    except:
        return False

def optimize_ollama():
    """Start Ollama with optimized settings if not running"""
    if check_ollama_status():
        print("✅ Ollama is already running")
        return True
    
    print("🚀 Starting Ollama with optimized GPU settings...")
    
    # Set environment variables for maximum GPU utilization
    env_vars = {
        "OLLAMA_NUM_PARALLEL": "8",           # Allow 8 parallel requests
        "OLLAMA_MAX_LOADED_MODELS": "1",      # Keep only 1 model loaded
        "OLLAMA_FLASH_ATTENTION": "1",        # Enable flash attention
        "OLLAMA_GPU_OVERHEAD": "0",           # Minimize GPU overhead
        "CUDA_VISIBLE_DEVICES": "0",          # Use first GPU
        "OLLAMA_MAX_QUEUE": "512",            # Increase queue size
    }
    
    try:
        # Set environment variables and start Ollama
        env = os.environ.copy()
        env.update(env_vars)
        
        print("🔧 Optimized Ollama settings:")
        for key, value in env_vars.items():
            print(f"   {key}={value}")
        
        # Start Ollama serve in background
        subprocess.Popen(["ollama", "serve"], env=env)
        
        # Wait for Ollama to be ready
        print("⏳ Waiting for Ollama to be ready...")
        for i in range(30):
            if check_ollama_status():
                print("✅ Ollama is ready!")
                return True
            time.sleep(1)
        
        print("❌ Ollama failed to start")
        return False
        
    except Exception as e:
        print(f"❌ Failed to start Ollama: {e}")
        return False

def start_server():
    """Start the FastAPI server with optimized settings"""
    print("🎯 Starting FlowHCM RAG API Server with Maximum Performance...")
    
    # Change to the project directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    # Step 1: Optimize Ollama
    if not optimize_ollama():
        print("❌ Failed to optimize Ollama")
        sys.exit(1)
    
    # Step 2: Set optimized environment variables for the API server
    env = os.environ.copy()
    env.update({
        # Python optimizations
        "PYTHONUNBUFFERED": "1",              # Unbuffered output
        "PYTHONDONTWRITEBYTECODE": "1",       # Don't write .pyc files
        "OMP_NUM_THREADS": "8",               # OpenMP threads
        "MKL_NUM_THREADS": "8",               # Intel MKL threads
        
        # CUDA optimizations
        "CUDA_LAUNCH_BLOCKING": "0",          # Async CUDA operations
        "CUDA_CACHE_DISABLE": "0",            # Enable CUDA cache
        
        # Memory optimizations
        "MALLOC_TRIM_THRESHOLD_": "100000",   # Memory trimming
    })
    
    print("🔧 Optimized API server settings:")
    print("   Workers: 7")
    print("   Host: 0.0.0.0")
    print("   Port: 8000")
    print("   Python optimizations: Enabled")
    print("   CUDA optimizations: Enabled")
    
    # Command to start uvicorn with optimized settings
    cmd = [
        sys.executable, "-m", "uvicorn",
        "apps.api:app",
        "--host", "0.0.0.0",
        "--port", "8000",
        "--workers", "7",
        "--access-log",                       # Enable access logging
    ]
    
    try:
        print("🚀 Launching optimized API server...")
        # Start the server with optimized environment
        subprocess.run(cmd, env=env, check=True)
    except KeyboardInterrupt:
        print("\n⏹️  Server stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error starting server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    start_server()