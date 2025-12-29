# Username-Based Session Management Implementation

## Problem
The previous implementation was creating a new session for every query, even when the same username was making requests. This led to:
- Unnecessary session proliferation
- Loss of conversation context between requests from the same user
- Poor user experience with fragmented conversations

## Solution
Implemented username-based session management that:
1. **Reuses existing active sessions** for the same username within the same client
2. **Creates new sessions only when needed** (no active session or session expired)
3. **Properly handles session expiry** based on configurable timeout
4. **Supports multiple users per client** with separate sessions per username

## Key Changes

### 1. Database Layer (`src/database/postgres_manager.py`)
- Added `get_active_session_for_user()` method to find existing active sessions by username and client
- Enhanced `create_session()` method with better logging
- Added `get_user_session_history()` method for session tracking
- Added `cleanup_expired_sessions_by_username()` for database cleanup

### 2. Session Manager (`src/fastapi_session_manager.py`)
- Updated `Session` dataclass to include `username` and `client_id`
- Added `find_active_session_for_user()` method for in-memory session lookup
- Added `get_or_create_session_for_user()` method for smart session management
- Enhanced session creation to track user information

### 3. API Layer (`apps/api.py`)
- Modified `/query` endpoint to check for existing active sessions before creating new ones
- Added username-based session lookup logic
- Enhanced session cleanup task to clean both memory and database
- Added `/sessions/user/{username}` endpoint for user session history

## Session Flow

### New Session Creation Flow:
1. **Check Database**: Look for existing active session for username + client
2. **Check Memory**: If found in DB, try to restore to memory
3. **Reuse or Create**: Either reuse existing session or create new one
4. **Update Browser**: Sync browser session cookie with current session

### Session Reuse Logic:
```
User makes request with username "john_doe"
├── Check database for active session (last_active < SESSION_MAX_AGE)
├── If found: Check if session exists in memory
│   ├── If in memory: Reuse existing session
│   └── If not in memory: Restore session to memory from DB
└── If not found: Create completely new session
```

## Configuration
- `SESSION_MAX_AGE`: Maximum session age in seconds (default: 1800 = 30 minutes)
- Sessions expire based on `last_active` timestamp
- Cleanup runs every `CLEANUP_INTERVAL` seconds

## Benefits
1. **Conversation Continuity**: Users maintain context across requests
2. **Resource Efficiency**: Fewer unnecessary sessions created
3. **Multi-User Support**: Each username gets its own session per client
4. **Proper Expiry**: Sessions expire based on inactivity, not arbitrary creation
5. **Scalability**: Supports multiple clients with multiple users each

## API Usage
The session management is transparent to API consumers. Simply include the `X-User-Name` header:

```bash
curl -X POST "http://localhost:8000/query" \
  -H "X-Company-Pin: YOUR_PIN" \
  -H "X-API-Key: YOUR_KEY" \
  -H "X-User-Name: john_doe" \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the leave policy?"}'
```

## Testing
Run the test script to verify functionality:
```bash
python test_session_management.py
```

## Monitoring
- Check active sessions: `GET /sessions`
- Check user session history: `GET /sessions/user/{username}`
- Monitor logs for session creation/reuse messages