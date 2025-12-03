# Token Limiting System

## Overview
The chatbot now includes a token limiting system to control usage per client on a monthly basis.

## Database Schema

### 1. Clients Table
- **token_limit_per_month**: Monthly token limit (default: 100,000)

### 2. Tokens Table (New)
Tracks monthly token usage per client:
- **fk_client_id**: Reference to client
- **month_year**: Format "dec_2025" (month_year)
- **tokens_used**: Total tokens used in that month
- **created_at**: First usage timestamp
- **updated_at**: Last update timestamp

### 3. Conversation Tables
- **tokens_used**: Tokens consumed by this specific query (input + output)

## How It Works

### 1. Token Counting
Uses `tiktoken` library (same as OpenAI) to count tokens:
- **Input tokens**: User query
- **Output tokens**: Bot response
- **Total tokens**: Input + Output

If tiktoken is not available, falls back to approximate counting (words / 0.75).

### 2. Flow

**On each /query request:**

1. **Authenticate client** ✅
2. **Check token limit**:
   - Get client's `token_limit_per_month`
   - Get current month's usage from `tokens` table
   - If `usage >= limit` → Return "Token limit exceeded" message
3. **Process query** ✅
4. **Count tokens** (input + output)
5. **Save conversation** with `tokens_used` column
6. **Update token usage** in `tokens` table (increment)
7. **Return response** ✅

### 3. Monthly Reset
Token usage is tracked per month using `month_year` format (e.g., "dec_2025").

When a new month starts:
- New row is automatically created in `tokens` table
- Previous month's data is preserved for history

No manual reset needed - the system automatically starts fresh each month.

## Installation

### 1. Install tiktoken
```bash
pip install tiktoken
```

### 2. Run Migration
```bash
psql -h localhost -U postgres -d chatbot_db -f migration_add_token_limits.sql
```

This will:
- Add `token_limit_per_month` column to `clients` table
- Create `tokens` table for monthly tracking
- Add `tokens_used` column to all existing conversation tables

## Configuration

### Set Token Limit for a Client
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

### Check Token Usage
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
```

### View Token Usage Per Conversation
```sql
-- See tokens used in each conversation
SELECT 
    conversation_id,
    user_message,
    tokens_used,
    created_at
FROM chatbot.conversation_dec_2025
WHERE fk_session_id = 123
ORDER BY created_at DESC;
```

## API Response When Limit Exceeded

When a client exceeds their monthly token limit:

```json
{
  "response": "Your monthly token limit has been reached. Please contact your administrator.",
  "session_id": ""
}
```

## Logging

Token usage is logged in:
1. **Main log**: `logs/chatbot_YYYY-MM-DD.log`
2. **Client log**: `logs/{company_pin}_YYYY-MM-DD.log`

Example log output:
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

## PostgreSQL Manager Methods

### `get_client_token_limit(client_id)`
Returns the monthly token limit for a client.

### `get_client_token_usage(client_id, month_year=None)`
Returns total tokens used by client in specified month (defaults to current month).

### `update_token_usage(client_id, tokens_used, month_year=None)`
Adds tokens to client's monthly usage.

### `check_token_limit(client_id)`
Returns dict with:
- `allowed`: Boolean (can make request?)
- `usage`: Current usage
- `limit`: Monthly limit
- `remaining`: Tokens remaining

## Future Enhancements (Not Implemented Yet)

These features are planned but not yet implemented:

1. **Admin Panel in HCMSAPI**:
   - View token usage per client
   - Set custom limits per client
   - View usage history
   - Manual reset option

2. **Usage Analytics**:
   - Daily usage trends
   - Peak usage times
   - Cost estimation

3. **Alerts**:
   - Email when 80% limit reached
   - Notify admin when client hits limit

4. **Rate Limiting**:
   - Requests per minute/hour
   - Concurrent request limits

## Testing

Test token limiting:

```bash
# 1. Set a low limit for testing
psql -h localhost -U postgres -d chatbot_db -c "UPDATE chatbot.clients SET token_limit_per_month = 100 WHERE client_id = 1;"

# 2. Make requests until limit is reached
# Use Postman or curl to send queries

# 3. Check usage
psql -h localhost -U postgres -d chatbot_db -c "SELECT * FROM chatbot.tokens WHERE fk_client_id = 1;"

# 4. Reset for normal use
psql -h localhost -U postgres -d chatbot_db -c "UPDATE chatbot.clients SET token_limit_per_month = 100000 WHERE client_id = 1;"
```

## Notes

- Default limit: **100,000 tokens/month**
- Token counting uses OpenAI's tiktoken (same as GPT models)
- Usage resets automatically each month
- Historical data is preserved in `tokens` table
- Each conversation stores its token cost in `tokens_used` column
