#!/usr/bin/env python3
"""
Script to run the sequential HR questions test
"""
import subprocess
import sys
import os
from datetime import datetime

def run_sequential_test():
    """Run the sequential HR questions test"""
    print("🧪 Starting Sequential HR Questions Test...")
    print(f"📅 Test started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Change to the project directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    # Command to run the sequential test
    cmd = [sys.executable, "test_sequential_users.py"]
    
    try:
        # Run the test
        result = subprocess.run(cmd, check=True, capture_output=False)
        print(f"\n✅ Test completed successfully!")
        print(f"📅 Test finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
    except KeyboardInterrupt:
        print("\n⏹️  Test stopped by user")
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Test failed with error: {e}")
        print("Check the logs above for details")
        sys.exit(1)
        
    except FileNotFoundError:
        print("❌ Error: test_sequential_users.py not found")
        print("Make sure you're running this script from the project root directory")
        sys.exit(1)

if __name__ == "__main__":
    run_sequential_test()