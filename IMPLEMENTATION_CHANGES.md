# Chatbot Authentication & Database Implementation Changes

## Overview
This document outlines the changes made to implement proper authentication and real-time database persistence for the FlowHCM chatbot.

---

## 1. Database Schema Changes

### New Schema Structure

#### `chatbot.clients` Table
```sql
CREATE TABLE chatbot.clients (
    client_id INTEGER PRIMARY KEY,           -- From FlowHCM
    company_pin VARCHAR(50) NOT NULL,        -- Company PIN for auth
    api_key VARCHAR(255) UNIQUE NOT NULL,    -- Generated API key
    is_active BOOLEAN DEFAULT TRUE,          -- Active status
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

**Purpose**: Store client authentication credentials synced from HCMSAPI.

#### `chatbot.sessions` Table
```sql
CREATE TABLE chatbot.sessions (
    session_id UUID PRIMARY KEY,             -- Session identifier
    username VARCHAR(255) NOT NULL,          -- FlowHCM username
    fk_client_id INTEGER NOT NULL,           -- Foreign key to clients
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_active TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (fk_client_id) REFERENCES chatbot.clients(client_id)
);
```

**Purpose**: Store session metadata (who, when, which client).

#### `chatbot.conversation` Table
```sql
CREATE TABLE chatbot.conversation (
    conversation_id SERIAL PRIMARY KEY,      -- Auto-increment ID
    fk_session_id UUID NOT NULL,             -- Foreign key to sessions
    user_message TEXT NOT NULL,              -- User's question
    chatbot_response TEXT NOT NULL,          -- Bot's answer
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (fk_session_id) REFERENCES chatbot.sessions(session_id)
);
```

**Purpose**: Store individual message-response pairs in real-time.

### Key Changes from Old Schema
- ✅ Split sessions into 2 tables (sessions + conversation)
- ✅ Changed clients table structure (removed client_name, added company_pin)
- ✅ Removed JSONB messages column (now using conversation table)
- ✅ Added username to sessions table
- ✅ One row per message-response pair in conversation table

---

## 2. PostgreSQL Manager Updates

### New Methods

#### `authenticate_client(client_id, company_pin, api_key)`
**Purpose**: Validate client credentials against all three parameters.

**Returns**: Client info dict if authenticated, None if failed.

**Usage**:
```python
client = db.authenticate_client(1, "1032", "FLOW-1-abc123...")
if client:
    # Authenticated
else:
    # Unauthorized
```

#### `sync_client(client_id, company_pin, api_key, is_active)`
**Purpose**: Sync client data from HCMSAPI (UPSERT operation).

**Called**: When API key is generated in HCMSAPI.

**Usage**:
```python
db.sync_client(1, "1032", "FLOW-1-abc123...", True)
```

#### `create_session(session_id, username, client_id)`
**Purpose**: Create new session record with user and client info.

**Usage**:
```python
db.create_session(uuid.uuid4(), "admin", 1)
```

#### `add_conversation(session_id, user_message, chatbot_response)`
**Purpose**: Add message-response pair to conversation table in real-time.

**Usage**:
```python
db.add_conversation(session_id, "What is my leave balance?", "Your leave balance is 15 days.")
```

#### `update_session_activity(session_id)`
**Purpose**: Update last_active timestamp for session.

**Usage**:
```python
db.update_session_activity(session_id)
```

#### `get_session_conversations(session_id, limit)`
**Purpose**: Retrieve conversation history for a session.

**Returns**: List of conversation dictionaries.

---

## 3. FastAPI Changes

### Authentication Flow

#### Request Headers Required
```
X-Client-ID: 1
X-Company-Pin: 1032
X-API-Key: FLOW-1-abc123...
X-User-Name: admin
```

#### Authentication Process
```python
1. Extract headers from request
2. Validate all required headers present
3. Connect to PostgreSQL
4. Call db.authenticate_client(client_id, company_pin, api_key)
5. If authentication fails → Return "Unauthorized"
6. If authentication succeeds → Process query
```

### `/query` Endpoint Changes

**Before**:
- No authentication
- Only session cookies

**After**:
- ✅ Validates X-Client-ID, X-Company-Pin, X-API-Key headers
- ✅ Authenticates against PostgreSQL clients table
- ✅ Returns "Unauthorized" if auth fails
- ✅ Creates session in PostgreSQL on first query
- ✅ Saves each conversation to PostgreSQL in real-time
- ✅ Updates session activity timestamp

### New `/admin/sync-client` Endpoint

**Purpose**: Sync client data from HCMSAPI to PostgreSQL.

**Method**: POST

**Request Body**:
```json
{
  "client_id": 1,
  "company_pin": "1032",
  "api_key": "FLOW-1-abc123...",
  "is_active": true
}
```

**Response**:
```json
{
  "success": true,
  "message": "Client 1 synced successfully"
}
```

**Called By**: HCMSAPI ChatbotBusiness.SyncToPostgreSQL() method.

---

## 4. Real-Time Database Persistence

### When Data is Saved

#### Session Creation
**Trigger**: First query from a new session
**Action**: Insert into `chatbot.sessions` table
```sql
INSERT INTO chatbot.sessions (session_id, username, fk_client_id, created_at, last_active)
VALUES (uuid, 'admin', 1, NOW(), NOW());
```

#### Each Query-Response
**Trigger**: Every user query and bot response
**Action**: Insert into `chatbot.conversation` table
```sql
INSERT INTO chatbot.conversation (fk_session_id, user_message, chatbot_response, created_at)
VALUES (uuid, 'What is my leave balance?', 'Your leave balance is 15 days.', NOW());
```

#### Session Activity
**Trigger**: Every query
**Action**: Update `chatbot.sessions.last_active`
```sql
UPDATE chatbot.sessions SET last_active = NOW() WHERE session_id = uuid;
```

---

## 5. Integration with HCMSAPI

### HCMSAPI Flow

#### When API Key is Generated
```csharp
1. ChatbotBusiness.GenerateAPIKeyForClient(clientId, createdBy)
2. Generate secure API key using HMACSHA256
3. Save to MySQL (FlowHCM database)
4. Get client info (CompanyPin)
5. Call SyncToPostgreSQL() async
   → POST to FastAPI /admin/sync-client
   → Saves to PostgreSQL chatbot.clients table
```

#### When User Sends Query
```csharp
1. ChatbotController.GetResponse(usermsg)
2. Extract token from headers
3. Get UserData from memCache (ClientID, UserName)
4. Get client info from MySQL (APIKey, CompanyPin)
5. Call ChatbotAPI() with headers:
   - X-Client-ID
   - X-Company-Pin
   - X-API-Key
   - X-User-Name
6. FastAPI authenticates and processes
7. Return response to frontend
```

---

## 6. Testing Checklist

### Database Setup
- [ ] Run updated schema.sql
- [ ] Verify tables created: clients, sessions, conversation
- [ ] Verify indexes created
- [ ] Insert test client (optional)

### HCMSAPI Integration
- [ ] Generate API key for test client
- [ ] Verify key saved to MySQL
- [ ] Verify key synced to PostgreSQL clients table
- [ ] Check company_pin populated correctly

### Authentication Testing
- [ ] Send query with valid credentials → Success
- [ ] Send query with invalid API key → "Unauthorized"
- [ ] Send query with wrong company_pin → "Unauthorized"
- [ ] Send query with missing headers → "Unauthorized"
- [ ] Send query with inactive client → "Unauthorized"

### Database Persistence Testing
- [ ] Send first query → Session created in PostgreSQL
- [ ] Send query → Conversation saved to PostgreSQL
- [ ] Send multiple queries → Multiple conversation rows
- [ ] Check last_active updates on each query
- [ ] Verify username and client_id in sessions table

### Query Conversation History
```sql
-- Get all conversations for a session
SELECT * FROM chatbot.conversation 
WHERE fk_session_id = 'your-session-uuid'
ORDER BY created_at;

-- Get all sessions for a client
SELECT * FROM chatbot.sessions 
WHERE fk_client_id = 1
ORDER BY last_active DESC;

-- Get conversation count per session
SELECT s.session_id, s.username, COUNT(c.conversation_id) as msg_count
FROM chatbot.sessions s
LEFT JOIN chatbot.conversation c ON c.fk_session_id = s.session_id
WHERE s.fk_client_id = 1
GROUP BY s.session_id, s.username;
```

---

## 7. Migration Steps

### Step 1: Backup Existing Data
```bash
pg_dump -U postgres chatbot > chatbot_backup_before_migration.sql
```

### Step 2: Drop Old Tables (if needed)
```sql
DROP TABLE IF EXISTS chatbot.sessions CASCADE;
DROP TABLE IF EXISTS chatbot.clients CASCADE;
```

### Step 3: Run New Schema
```bash
psql -U postgres -d chatbot -f src/database/schema.sql
```

### Step 4: Restart FastAPI
```bash
python apps/api.py
```

### Step 5: Regenerate API Keys
- Go to HCMSAPI admin panel
- Regenerate API keys for all clients
- This will sync them to PostgreSQL

### Step 6: Test Authentication
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -H "X-Client-ID: 1" \
  -H "X-Company-Pin: 1032" \
  -H "X-API-Key: FLOW-1-abc123..." \
  -H "X-User-Name: admin" \
  -d '{"query": "What is FlowHCM?"}'
```

---

## 8. Key Benefits

### Security
✅ Multi-factor authentication (client_id + company_pin + api_key)
✅ Hardcoded "Unauthorized" response for failed auth
✅ No sensitive data in error messages

### Data Persistence
✅ Real-time conversation storage
✅ Complete audit trail
✅ Easy to query conversation history
✅ Session tracking per user and client

### Scalability
✅ Normalized database structure
✅ Efficient queries with proper indexes
✅ Separate tables for sessions and conversations

### Analytics
✅ Track usage per client
✅ Track usage per user
✅ Analyze conversation patterns
✅ Monitor session activity

---

## 9. Configuration

### Environment Variables (.env)
```properties
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=chatbot
POSTGRES_SCHEMA=chatbot
POSTGRES_USER=postgres
POSTGRES_PASSWORD=root
```

### HCMSAPI Web.config
```xml
<add key="FastAPIURL" value="http://localhost:8000" />
<add key="APIKeySecret" value="FlowHCM-Secret-Key-2025" />
```

---

## 10. Troubleshooting

### "Unauthorized" Response
**Check**:
1. Client exists in PostgreSQL clients table
2. API key matches exactly
3. Company PIN matches exactly
4. Client is_active = TRUE
5. All headers present in request

### Session Not Created
**Check**:
1. PostgreSQL connection successful
2. Username provided in X-User-Name header
3. Check PostgreSQL logs for errors

### Conversation Not Saved
**Check**:
1. Session exists in sessions table
2. Foreign key constraint satisfied
3. Check PostgreSQL logs for errors

---

## Document Version
- **Version**: 1.0
- **Date**: December 2, 2025
- **Author**: FlowHCM Development Team
