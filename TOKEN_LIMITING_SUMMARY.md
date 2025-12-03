# Token Limiting Implementation - Quick Summary

## What Was Implemented

✅ **Token limiting system** to control API usage per client per month

## Database Changes

### 1. Clients Table
- Added `token_limit_per_month` column (default: 100,000)
- Removed old `n_tokens` column

### 2. New Tokens Table
Tracks monthly usage per client:
- `fk_client_id` - Client reference
- `month_year` - Format: "dec_2025"
- `tokens_used` - Total tokens used this month
- Auto-creates new row each month

### 3. Conversation Tables
- Added `tokens_used` column to track per-query token cost

## How It Works

### Flow
1. **Authenticate** client (company_pin + api_key)
2. **Check limit**: Get usage from `tokens` table
3. **Block if exceeded**: Return "Token limit exceeded" message
4. **Process query** if allowed
5. **Count tokens**: Input + Output using tiktoken
6. **Save conversation** with token count
7. **Update usage**: Increment in `tokens` table

### Token Counting
- Uses `tiktoken` library (same as OpenAI)
- Counts input tokens (user query)
- Counts output tokens (bot response)
- Total = input + output
- Fallback to approximate if tiktoken unavailable

### Monthly Reset
- Automatic - no manual intervention needed
- New month = new row in `tokens` table
- Previous months preserved for history

## Installation

```bash
# 1. Install tiktoken
pip install tiktoken

# 2. Run migration
psql -h localhost -U postgres -d chatbot_db -f migration_add_token_limits.sql

# 3. Restart API
python apps/api.py
```

## Configuration

```sql
-- Set limit for a client
UPDATE chatbot.clients 
SET token_limit_per_month = 50000 
WHERE client_id = 1;

-- Check usage
SELECT * FROM chatbot.tokens 
WHERE fk_client_id = 1 
AND month_year = 'dec_2025';
```

## Testing

```bash
# Test token counting
python test_token_counting.py

# Test with low limit
psql -c "UPDATE chatbot.clients SET token_limit_per_month = 100 WHERE client_id = 1;"
# Make requests until blocked
# Check usage in tokens table
```

## Files Changed

1. `src/database/schema.sql` - Schema updates
2. `src/database/postgres_manager.py` - Token methods
3. `apps/api.py` - Token checking in /query
4. `src/utils.py` - Token counting functions
5. `requirements.txt` - Added tiktoken

## Files Created

1. `migration_add_token_limits.sql` - Migration script
2. `TOKEN_LIMITING_GUIDE.md` - Full documentation
3. `test_token_counting.py` - Test script
4. `TOKEN_LIMITING_SUMMARY.md` - This file

## Default Settings

- **Default limit**: 100,000 tokens/month
- **Limit exceeded message**: "Your monthly token limit has been reached. Please contact your administrator."
- **Automatic reset**: First day of each month

## Logging

Token usage logged in:
- Main log: `logs/chatbot_YYYY-MM-DD.log`
- Client log: `logs/{company_pin}_YYYY-MM-DD.log`

Shows:
- Token limit check (usage, limit, remaining)
- Token counting (input, output, total)
- Token usage update (new total)

## Next Steps (Not Implemented)

Future enhancements:
- Admin panel in HCMSAPI to manage limits
- Usage analytics dashboard
- Email alerts at 80% usage
- Rate limiting (requests per minute)

## Support

See `TOKEN_LIMITING_GUIDE.md` for detailed documentation.
