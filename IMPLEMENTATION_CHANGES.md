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


---

## 11. Token Limiting System (NEW)

### Overview
Implemented token-based usage control to limit API consumption per client on a monthly basis.

### Database Schema Changes

#### Updated `chatbot.clients` Table
```sql
ALTER TABLE chatbot.clients 
ADD COLUMN token_limit_per_month INTEGER DEFAULT 100000;
```

**Purpose**: Store monthly token limit for each client (default: 100,000 tokens).

#### New `chatbot.tokens` Table
```sql
CREATE TABLE chatbot.tokens (
    token_id SERIAL PRIMARY KEY,
    fk_client_id INTEGER NOT NULL,
    month_year VARCHAR(10) NOT NULL,              -- Format: "dec_2025"
    tokens_used INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (fk_client_id) REFERENCES chatbot.clients(client_id) ON DELETE CASCADE,
    UNIQUE(fk_client_id, month_year)
);
```

**Purpose**: Track monthly token usage per client with historical data.

#### Updated `chatbot.conversation_*` Tables
```sql
ALTER TABLE chatbot.conversation_dec_2025 
ADD COLUMN tokens_used INTEGER DEFAULT 0;
```

**Purpose**: Store token count for each individual query-response pair.

### Token Counting

#### Library: tiktoken
```bash
pip install tiktoken
```

**Why tiktoken?**
- Same tokenizer used by OpenAI GPT models
- Accurate token counting for cost estimation
- Fallback to approximate method if not available

#### Token Calculation
```python
input_tokens = count_tokens(user_query)
output_tokens = count_tokens(bot_response)
total_tokens = input_tokens + output_tokens
```

### Authentication Flow with Token Limiting

#### Updated `/query` Endpoint Flow
```python
1. Authenticate client (client_id + company_pin + api_key) ✅
2. Check token limit:
   - Get client's token_limit_per_month
   - Get current month's usage from tokens table
   - If usage >= limit → Return "Token limit exceeded"
3. Process query ✅
4. Count tokens (input + output)
5. Save conversation with tokens_used column
6. Update tokens table (increment usage)
7. Return response ✅
```

### PostgreSQL Manager Methods

#### `get_client_token_limit(client_id)`
Returns the monthly token limit for a client.

```python
limit = db.get_client_token_limit(1)  # Returns 100000
```

#### `get_client_token_usage(client_id, month_year=None)`
Returns total tokens used by client in specified month (defaults to current month).

```python
usage = db.get_client_token_usage(1)  # Returns current month usage
usage = db.get_client_token_usage(1, "dec_2025")  # Returns specific month
```

#### `update_token_usage(client_id, tokens_used, month_year=None)`
Adds tokens to client's monthly usage (UPSERT operation).

```python
db.update_token_usage(1, 175)  # Add 175 tokens to current month
```

#### `check_token_limit(client_id)`
Returns dict with usage status.

```python
status = db.check_token_limit(1)
# Returns:
# {
#     'allowed': True,
#     'usage': 45230,
#     'limit': 100000,
#     'remaining': 54770
# }
```

### Monthly Reset Behavior

**Automatic Reset**: Token usage is tracked per month using `month_year` format.

When a new month starts:
- New row is automatically created in `tokens` table
- Previous month's data is preserved for history
- No manual reset needed

Example:
```sql
-- December 2025
fk_client_id | month_year | tokens_used
1            | dec_2025   | 45230

-- January 2026 (auto-created on first query)
fk_client_id | month_year | tokens_used
1            | jan_2026   | 0
```

### API Response When Limit Exceeded

```json
{
  "response": "Your monthly token limit has been reached. Please contact your administrator.",
  "session_id": ""
}
```

### Logging

Token usage is logged in both main and client-specific logs:

```
🔢 TOKEN LIMIT CHECK
   Token Limit: 100,000
   Tokens Used: 45,230
   Remaining: 54,770
   Allowed: True

🔢 TOKEN COUNTING
   Input tokens: 25
   Output tokens: 150
   Total tokens: 175

✅ Token usage updated
   Client ID: 1
   Tokens Added: 175
   New Total Usage: 45,405 / 100,000
   Remaining: 54,595
```

### Migration

#### Run Migration Script
```bash
psql -h localhost -U postgres -d chatbot_db -f migration_add_token_limits.sql
```

This will:
- Add `token_limit_per_month` column to `clients` table
- Create `tokens` table for monthly tracking
- Add `tokens_used` column to all existing conversation tables
- Create necessary indexes

### Configuration

#### Set Token Limit for a Client
```sql
-- Set limit to 50,000 tokens per month
UPDATE chatbot.clients 
SET token_limit_per_month = 50000 
WHERE client_id = 1;

-- Set unlimited (very high limit)
UPDATE chatbot.clients 
SET token_limit_per_month = 999999999 
WHERE client_id = 2;
```

#### Check Token Usage
```sql
-- Current month usage for a client
SELECT * FROM chatbot.tokens 
WHERE fk_client_id = 1 
AND month_year = 'dec_2025';

-- All-time usage for a client
SELECT 
    month_year,
    tokens_used,
    updated_at
FROM chatbot.tokens 
WHERE fk_client_id = 1 
ORDER BY month_year DESC;

-- Token usage per conversation
SELECT 
    conversation_id,
    user_message,
    tokens_used,
    created_at
FROM chatbot.conversation_dec_2025
WHERE fk_session_id = 123
ORDER BY created_at DESC;
```

### Testing Token Limiting

#### Test Script
```bash
python test_token_counting.py
```

This will:
- Test token counting with various inputs
- Show difference between tiktoken and approximate counting
- Simulate conversation scenarios
- Project monthly usage and costs

#### Manual Testing
```bash
# 1. Set a low limit for testing
psql -c "UPDATE chatbot.clients SET token_limit_per_month = 100 WHERE client_id = 1;"

# 2. Make requests until limit is reached
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -H "X-Client-ID: 1" \
  -H "X-Company-Pin: 1032" \
  -H "X-API-Key: FLOW-1-abc123..." \
  -H "X-User-Name: admin" \
  -d '{"query": "What is FlowHCM?"}'

# 3. Check usage
psql -c "SELECT * FROM chatbot.tokens WHERE fk_client_id = 1;"

# 4. Reset for normal use
psql -c "UPDATE chatbot.clients SET token_limit_per_month = 100000 WHERE client_id = 1;"
```

### Key Benefits

#### Cost Control
✅ Prevent unlimited API usage
✅ Set different limits per client
✅ Track usage per month with history

#### Transparency
✅ Detailed token counting per query
✅ Monthly usage tracking
✅ Historical data preserved

#### Flexibility
✅ Configurable limits per client
✅ Automatic monthly reset
✅ No manual intervention needed

### Future Enhancements (Planned)

These features are planned but not yet implemented:

1. **Admin Panel in HCMSAPI**:
   - View token usage per client
   - Set custom limits per client
   - View usage history
   - Manual reset option

2. **Usage Analytics**:
   - Daily usage trends
   - Peak usage times
   - Cost estimation dashboard

3. **Alerts**:
   - Email when 80% limit reached
   - Notify admin when client hits limit

4. **Rate Limiting**:
   - Requests per minute/hour
   - Concurrent request limits

### Files Modified

- `src/database/schema.sql` - Added token_limit_per_month, tokens table, tokens_used column
- `src/database/postgres_manager.py` - Added token management methods
- `apps/api.py` - Added token checking and counting in /query endpoint
- `src/utils.py` - Added count_tokens() and approximate_token_count() functions
- `requirements.txt` - Added tiktoken>=0.5.0

### New Files Created

- `migration_add_token_limits.sql` - Migration script for existing databases
- `TOKEN_LIMITING_GUIDE.md` - Comprehensive guide for token limiting system
- `test_token_counting.py` - Test script for token counting functionality

---

## Document Version
- **Version**: 2.0
- **Date**: December 3, 2025
- **Author**: FlowHCM Development Team
- **Changes**: Added Token Limiting System documentation
