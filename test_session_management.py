#!/usr/bin/env python3
"""
Test script for username-based session management
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from fastapi_session_manager import FastAPISessionManager
from database.postgres_manager import PostgresManager
from config import POSTGRES_CONNECTION_STRING, SESSION_MAX_AGE
import time

def test_session_management():
    """Test the new username-based session management"""
    print("🧪 Testing Username-Based Session Management")
    print("=" * 50)
    
    # Initialize session manager
    session_manager = FastAPISessionManager(max_sessions=100, session_max_age=SESSION_MAX_AGE)
    
    # Test 1: Create session for user1
    print("\n1️⃣ Creating session for user1...")
    session1 = session_manager.create_session("user1", 1)
    print(f"   Session ID: {session1.session_id}")
    print(f"   Username: {session1.username}")
    print(f"   Client ID: {session1.client_id}")
    
    # Test 2: Try to get session for same user (should reuse)
    print("\n2️⃣ Getting session for user1 again...")
    session1_reused = session_manager.get_or_create_session_for_user("user1", 1)
    print(f"   Session ID: {session1_reused.session_id}")
    print(f"   Same session? {session1.session_id == session1_reused.session_id}")
    
    # Test 3: Create session for different user
    print("\n3️⃣ Creating session for user2...")
    session2 = session_manager.create_session("user2", 1)
    print(f"   Session ID: {session2.session_id}")
    print(f"   Username: {session2.username}")
    print(f"   Different from user1? {session1.session_id != session2.session_id}")
    
    # Test 4: Create session for same user but different client
    print("\n4️⃣ Creating session for user1 on different client...")
    session3 = session_manager.create_session("user1", 2)
    print(f"   Session ID: {session3.session_id}")
    print(f"   Client ID: {session3.client_id}")
    print(f"   Different from user1/client1? {session1.session_id != session3.session_id}")
    
    # Test 5: List all sessions
    print("\n5️⃣ Listing all active sessions...")
    sessions = session_manager.list_sessions()
    for i, session in enumerate(sessions, 1):
        print(f"   {i}. {session['username']}@client{session['client_id']} - {session['session_id'][:8]}...")
    
    print(f"\n✅ Total sessions created: {len(sessions)}")
    print("🎉 Session management test completed!")

def test_database_integration():
    """Test database integration"""
    print("\n🗄️ Testing Database Integration")
    print("=" * 50)
    
    try:
        db = PostgresManager(POSTGRES_CONNECTION_STRING)
        if not db.connect():
            print("❌ Failed to connect to database")
            return
        
        print("✅ Database connection successful")
        
        # Test getting active session for user
        print("\n🔍 Testing get_active_session_for_user...")
        active_session = db.get_active_session_for_user("test_user", 1, SESSION_MAX_AGE)
        if active_session:
            print(f"   Found active session: {active_session['session_id']}")
        else:
            print("   No active session found (expected for new test)")
        
        db.disconnect()
        print("✅ Database test completed")
        
    except Exception as e:
        print(f"❌ Database test failed: {e}")

if __name__ == "__main__":
    test_session_management()
    test_database_integration()