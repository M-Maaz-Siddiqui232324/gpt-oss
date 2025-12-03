# Token Limiting Installation Checklist

## Prerequisites
- ✅ PostgreSQL database running
- ✅ Chatbot API running
- ✅ Python environment active

## Step-by-Step Installation

### Step 1: Install tiktoken
```bash
pip install tiktoken
```

**Verify installation:**
```bash
python -c "import tiktoken; print('tiktoken installed successfully')"
```

### Step 2: Backup Database (Recommended)
```bash
pg_dump -h localhost -U postgres -d chatbot_db > backup_before_token_limiting.sql
```

### Step 3: Run Migration Script
```bash
psql -h localhost -U postgres -d chatbot_db -f migration_add_token_limits.sql
```

**Expected output:**
```
NOTICE:  Added token_limit_per_month column to clients table
NOTICE:  Removed old n_tokens column from clients table
CREATE TABLE
CREATE INDEX
NOTICE:  Added tokens_used column to conversation_dec_2025
 status                          | total_clients | clients_with_limits
---------------------------------+---------------+--------------------
 Migration completed successfully|             2 |                   2
```

### Step 4: Verify Database Changes

**Check clients table:**
```sql
\d chatbot.clients
```
Should show `token_limit_per_month` column.

**Check tokens table:**
```sql
\d chatbot.tokens
```
Should exist with columns: token_id, fk_client_id, month_year, tokens_used, created_at, updated_at.

**Check conversation table:**
```sql
\d chatbot.conversation_dec_2025
```
Should show `tokens_used` column.

### Step 5: Restart API Server
```bash
# Stop current server (Ctrl+C)

# Start server
python apps/api.py
```

**Check logs for:**
```
INFO - RAG system initialized successfully
INFO - Session manager initialized successfully
INFO - Starting server on 0.0.0.0:8000
```

### Step 6: Test Token Counting
```bash
python test_token_counting.py
```

**Expected output:**
```
================================================================================
TOKEN COUNTING TEST
================================================================================

Test: Short greeting
Text: Hello, how are you?
Length: 19 characters, 4 words
Tokens (tiktoken): 6
Tokens (approximate): 5
Difference: 1
--------------------------------------------------------------------------------
...
TEST COMPLETED
```

### Step 7: Test API with Token Limiting

**Test 1: Normal Query (Should Work)**
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -H "X-Company-Pin: 1032" \
  -H "X-API-Key: flowhcm_default_key_12345" \
  -H "X-User-Name: admin" \
  -d '{"query": "What is FlowHCM?"}'
```

**Expected response:**
```json
{
  "response": "FlowHCM is a comprehensive Human Capital Management system...",
  "session_id": "uuid-here"
}
```

**Check logs:**
```
🔢 TOKEN LIMIT CHECK
   Token Limit: 100,000
   Tokens Used: 0
   Remaining: 100,000
   Allowed: True

🔢 TOKEN COUNTING
   Input tokens: 4
   Output tokens: 150
   Total tokens: 154

✅ Token usage updated
   Client ID: 1
   Tokens Added: 154
   New Total Usage: 154 / 100,000
   Remaining: 99,846
```

**Test 2: Check Token Usage in Database**
```sql
SELECT * FROM chatbot.tokens WHERE fk_client_id = 1;
```

**Expected result:**
```
 token_id | fk_client_id | month_year | tokens_used |      created_at      |      updated_at
----------+--------------+------------+-------------+----------------------+----------------------
        1 |            1 | dec_2025   |         154 | 2025-12-03 10:30:00  | 2025-12-03 10:30:00
```

**Test 3: Test Token Limit (Set Low Limit)**
```sql
UPDATE chatbot.clients SET token_limit_per_month = 100 WHERE client_id = 1;
```

**Make query (should be blocked if usage > 100):**
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -H "X-Company-Pin: 1032" \
  -H "X-API-Key: flowhcm_default_key_12345" \
  -H "X-User-Name: admin" \
  -d '{"query": "What is FlowHCM?"}'
```

**Expected response (if limit exceeded):**
```json
{
  "response": "Your monthly token limit has been reached. Please contact your administrator.",
  "session_id": ""
}
```

**Reset limit:**
```sql
UPDATE chatbot.clients SET token_limit_per_month = 100000 WHERE client_id = 1;
```

### Step 8: Verify Logging

**Check main log:**
```bash
tail -f logs/chatbot_2025-12-03.log
```

**Check client-specific log:**
```bash
tail -f logs/1032_2025-12-03.log
```

Both should show token counting information.

## Configuration

### Set Custom Token Limits

**For specific client:**
```sql
UPDATE chatbot.clients 
SET token_limit_per_month = 50000 
WHERE client_id = 1;
```

**For all clients:**
```sql
UPDATE chatbot.clients 
SET token_limit_per_month = 75000;
```

**Unlimited (very high limit):**
```sql
UPDATE chatbot.clients 
SET token_limit_per_month = 999999999 
WHERE client_id = 1;
```

## Monitoring

### Check Current Usage
```sql
SELECT 
    c.client_id,
    c.company_pin,
    c.token_limit_per_month,
    COALESCE(t.tokens_used, 0) as current_usage,
    c.token_limit_per_month - COALESCE(t.tokens_used, 0) as remaining
FROM chatbot.clients c
LEFT JOIN chatbot.tokens t ON t.fk_client_id = c.client_id 
    AND t.month_year = 'dec_2025'
ORDER BY c.client_id;
```

### Check Usage History
```sql
SELECT 
    c.company_pin,
    t.month_year,
    t.tokens_used,
    c.token_limit_per_month,
    ROUND((t.tokens_used::float / c.token_limit_per_month * 100), 2) as usage_percent
FROM chatbot.tokens t
JOIN chatbot.clients c ON c.client_id = t.fk_client_id
ORDER BY t.month_year DESC, c.client_id;
```

### Check Per-Query Token Cost
```sql
SELECT 
    conversation_id,
    LEFT(user_message, 50) as query,
    tokens_used,
    created_at
FROM chatbot.conversation_dec_2025
WHERE fk_session_id = (
    SELECT id FROM chatbot.sessions 
    WHERE session_id = 'your-session-uuid'
)
ORDER BY created_at DESC
LIMIT 10;
```

## Troubleshooting

### Issue: "tiktoken not installed" warning
**Solution:**
```bash
pip install tiktoken
# Restart API server
```

### Issue: Migration fails with "column already exists"
**Solution:** Column already added, safe to ignore. Or drop and recreate:
```sql
ALTER TABLE chatbot.clients DROP COLUMN IF EXISTS token_limit_per_month;
-- Then run migration again
```

### Issue: Token usage not updating
**Check:**
1. PostgreSQL connection successful?
2. Check API logs for errors
3. Verify tokens table exists
4. Check foreign key constraints

**Debug query:**
```sql
SELECT * FROM chatbot.tokens WHERE fk_client_id = 1;
```

### Issue: Always getting "Token limit exceeded"
**Check current usage:**
```sql
SELECT * FROM chatbot.tokens 
WHERE fk_client_id = 1 
AND month_year = 'dec_2025';
```

**Reset usage (for testing):**
```sql
DELETE FROM chatbot.tokens 
WHERE fk_client_id = 1 
AND month_year = 'dec_2025';
```

### Issue: Tokens not counting accurately
**Check:**
1. tiktoken installed? `pip list | grep tiktoken`
2. Using approximate fallback? Check logs for warning
3. Test with: `python test_token_counting.py`

## Rollback (If Needed)

### Remove Token Limiting
```sql
-- Remove tokens table
DROP TABLE IF EXISTS chatbot.tokens CASCADE;

-- Remove token_limit_per_month column
ALTER TABLE chatbot.clients DROP COLUMN IF EXISTS token_limit_per_month;

-- Remove tokens_used from conversation tables
DO $$ 
DECLARE
    table_name TEXT;
BEGIN
    FOR table_name IN 
        SELECT tablename 
        FROM pg_tables 
        WHERE schemaname = 'chatbot' 
        AND tablename LIKE 'conversation_%'
    LOOP
        EXECUTE format('ALTER TABLE chatbot.%I DROP COLUMN IF EXISTS tokens_used', table_name);
    END LOOP;
END $$;
```

### Restore from Backup
```bash
psql -h localhost -U postgres -d chatbot_db < backup_before_token_limiting.sql
```

## Success Checklist

- [ ] tiktoken installed
- [ ] Database backup created
- [ ] Migration script executed successfully
- [ ] clients table has token_limit_per_month column
- [ ] tokens table created
- [ ] conversation tables have tokens_used column
- [ ] API server restarted
- [ ] Test script runs successfully
- [ ] API query works and counts tokens
- [ ] Token usage saved to database
- [ ] Logs show token counting information
- [ ] Token limit blocking works (tested with low limit)

## Next Steps

1. **Set appropriate limits** for each client based on their plan
2. **Monitor usage** regularly using SQL queries
3. **Review logs** to ensure token counting is accurate
4. **Plan for admin panel** to manage limits (future enhancement)

## Support

- Full documentation: `TOKEN_LIMITING_GUIDE.md`
- Flow diagram: `TOKEN_LIMITING_FLOW.txt`
- Quick summary: `TOKEN_LIMITING_SUMMARY.md`
- Implementation details: `IMPLEMENTATION_CHANGES.md`

## Questions?

Check the documentation files or review the code:
- Token counting: `src/utils.py` → `count_tokens()`
- Token management: `src/database/postgres_manager.py` → Token methods
- API integration: `apps/api.py` → `/query` endpoint
