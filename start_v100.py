#!/usr/bin/env python3
"""
Tesla V100 Optimized startup script for FlowHCM RAG API Server
Automatically configures and starts the server with maximum V100 performance
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

def stop_ollama():
    """Stop current Ollama instance"""
    print("🛑 Stopping current Ollama instance...")
    try:
        subprocess.run(["taskkill", "/f", "/im", "ollama.exe"], check=False, capture_output=True)
        time.sleep(3)
        print("✅ Ollama stopped")
    except:
        print("⚠️  Ollama was not running")

def start_v100_ollama():
    """Start Ollama with Tesla V100 optimizations"""
    print("🚀 Starting Ollama with Tesla V100 optimizations...")
    
    # Tesla V100 maximum performance settings
    v100_env = {
        "OLLAMA_NUM_PARALLEL": "12",          # Reduced for stability
        "OLLAMA_MAX_LOADED_MODELS": "1",      # Single model for max performance
        "OLLAMA_FLASH_ATTENTION": "1",        # Enable flash attention
        "OLLAMA_GPU_OVERHEAD": "0",           # Zero GPU overhead
        "CUDA_VISIBLE_DEVICES": "0",          # Tesla V100
        "OLLAMA_MAX_QUEUE": "1024",           # Reduced queue size
        "OLLAMA_BATCH_SIZE": "512",           # Reduced batch size
        "OLLAMA_CONTEXT_SIZE": "8192",        # Large context window
        "CUDA_LAUNCH_BLOCKING": "0",          # Async CUDA
        "CUDA_CACHE_DISABLE": "0",            # Enable CUDA cache
        "OLLAMA_KEEP_ALIVE": "24h",           # Keep model loaded
        "OLLAMA_NOPRUNE": "1",                # Don't prune model
    }
    
    try:
        env = os.environ.copy()
        env.update(v100_env)
        
        print("🔧 Tesla V100 Performance Settings:")
        for key, value in v100_env.items():
            print(f"   {key}={value}")
        
        # Start Ollama in background
        subprocess.Popen(["ollama", "serve"], env=env)
        
        # Wait for Ollama to be ready
        print("⏳ Waiting for Ollama to initialize...")
        for i in range(30):
            if check_ollama_status():
                print("✅ Ollama ready!")
                return True
            time.sleep(1)
        
        print("❌ Ollama failed to start")
        return False
        
    except Exception as e:
        print(f"❌ Failed to start Ollama: {e}")
        return False

def ensure_v100_model():
    """Ensure V100 optimized model exists and is loaded"""
    print("🔧 Ensuring V100 optimized model...")
    
    try:
        # Check if V100 model exists
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
        
        if "qwen3:8b-v100" not in result.stdout:
            print("📦 Creating V100 optimized model...")
            
            # Create V100 optimized modelfile
            modelfile = """FROM qwen3:8b

# Tesla V100 Performance Settings (Conservative)
PARAMETER num_ctx 4096
PARAMETER num_batch 512
PARAMETER num_gpu 99
PARAMETER num_thread 16
PARAMETER repeat_penalty 1.1
PARAMETER temperature 0.1
PARAMETER top_p 0.7
PARAMETER top_k 40
PARAMETER num_predict 400
"""
            
            with open("Modelfile.v100", "w") as f:
                f.write(modelfile)
            
            # Create the model
            create_result = subprocess.run(
                ["ollama", "create", "qwen3:8b-v100", "-f", "Modelfile.v100"],
                capture_output=True, text=True, timeout=300
            )
            
            if create_result.returncode != 0:
                print(f"❌ Failed to create V100 model: {create_result.stderr}")
                return False
            
            print("✅ V100 optimized model created")
        
        # Warm up the model
        print("🔥 Warming up V100 model...")
        subprocess.run(
            ["ollama", "run", "qwen3:8b-v100", "Hello"],
            capture_output=True, text=True, timeout=60
        )
        print("✅ V100 model warmed up and ready")
        
        return True
        
    except Exception as e:
        print(f"❌ Error with V100 model: {e}")
        return False

def update_config_for_v100():
    """Update config to use V100 optimized model"""
    print("⚙️  Configuring for Tesla V100...")
    
    try:
        config_path = "src/config.py"
        
        # Read current config
        with open(config_path, "r") as f:
            content = f.read()
        
        # Update to V100 model
        if 'MODEL_NAME = "qwen3:8b-v100"' not in content:
            content = content.replace(
                'MODEL_NAME = "qwen3:8b"',
                'MODEL_NAME = "qwen3:8b-v100"'
            )
            
            # Write updated config
            with open(config_path, "w") as f:
                f.write(content)
            
            print("✅ Config updated for V100 model")
        else:
            print("✅ Config already optimized for V100")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to update config: {e}")
        return False

def start_v100_api_server():
    """Start API server with V100 optimizations"""
    print("🚀 Starting Tesla V100 Optimized API Server...")
    
    # Change to project directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    # Tesla V100 optimized environment
    v100_api_env = {
        # Python optimizations
        "PYTHONUNBUFFERED": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "OMP_NUM_THREADS": "16",              # V100 has more cores
        "MKL_NUM_THREADS": "16",
        
        # CUDA optimizations for V100
        "CUDA_LAUNCH_BLOCKING": "0",
        "CUDA_CACHE_DISABLE": "0",
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        "CUDA_VISIBLE_DEVICES": "0",
        
        # Memory optimizations
        "MALLOC_TRIM_THRESHOLD_": "100000",
        
        # V100 specific
        "NVIDIA_VISIBLE_DEVICES": "0",
        "NVIDIA_DRIVER_CAPABILITIES": "compute,utility",
    }
    
    env = os.environ.copy()
    env.update(v100_api_env)
    
    print("🔧 Tesla V100 API Server Settings:")
    print("   Workers: 10 (optimized for V100)")
    print("   Host: 0.0.0.0")
    print("   Port: 8000")
    print("   Model: qwen3:8b-v100")
    print("   CUDA Optimizations: Enabled")
    print("   Memory Optimizations: Enabled")
    
    # Command with V100 optimizations
    cmd = [
        sys.executable, "-m", "uvicorn",
        "apps.api:app",
        "--host", "0.0.0.0",
        "--port", "8000",
        "--workers", "10",                    # More workers for V100
        "--access-log",
        "--log-level", "info",
    ]
    
    try:
        print("🎯 Launching Tesla V100 optimized server...")
        subprocess.run(cmd, env=env, check=True)
    except KeyboardInterrupt:
        print("\n⏹️  Server stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error starting server: {e}")
        sys.exit(1)

def main():
    """Main Tesla V100 startup function"""
    print("🎯 Tesla V100 Optimized FlowHCM RAG Server")
    print("=" * 50)
    
    # Step 1: Stop current Ollama
    stop_ollama()
    
    # Step 2: Start V100 optimized Ollama
    if not start_v100_ollama():
        print("❌ Failed to start V100 optimized Ollama")
        sys.exit(1)
    
    # Step 3: Ensure V100 model exists
    if not ensure_v100_model():
        print("❌ Failed to setup V100 model")
        sys.exit(1)
    
    # Step 4: Update config
    if not update_config_for_v100():
        print("❌ Failed to update config")
        sys.exit(1)
    
    # Step 5: Start API server
    print("\n🎉 Tesla V100 optimization complete!")
    print("🚀 Starting optimized API server...")
    start_v100_api_server()

if __name__ == "__main__":
    main()