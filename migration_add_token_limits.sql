-- Migration script to add token limiting features
-- Run this on existing database to add token tracking

-- 1. Add token_limit_per_month column to clients table (if not exists)
DO $$ 
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_schema = 'chatbot' 
        AND table_name = 'clients' 
        AND column_name = 'token_limit_per_month'
    ) THEN
        ALTER TABLE chatbot.clients 
        ADD COLUMN token_limit_per_month INTEGER DEFAULT 100000;
        RAISE NOTICE 'Added token_limit_per_month column to clients table';
    ELSE
        RAISE NOTICE 'token_limit_per_month column already exists';
    END IF;
END $$;

-- 2. Remove old n_tokens column if it exists
DO $$ 
BEGIN
    IF EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_schema = 'chatbot' 
        AND table_name = 'clients' 
        AND column_name = 'n_tokens'
    ) THEN
        ALTER TABLE chatbot.clients DROP COLUMN n_tokens;
        RAISE NOTICE 'Removed old n_tokens column from clients table';
    ELSE
        RAISE NOTICE 'n_tokens column does not exist';
    END IF;
END $$;

-- 3. Create tokens table for monthly tracking
CREATE TABLE IF NOT EXISTS chatbot.tokens (
    token_id SERIAL PRIMARY KEY,
    fk_client_id INTEGER NOT NULL,
    month_year VARCHAR(10) NOT NULL,
    tokens_used INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (fk_client_id) REFERENCES chatbot.clients(client_id) ON DELETE CASCADE,
    UNIQUE(fk_client_id, month_year)
);

-- 4. Create index on tokens table
CREATE INDEX IF NOT EXISTS idx_tokens_client_month 
ON chatbot.tokens(fk_client_id, month_year);

-- 5. Add tokens_used column to existing conversation tables
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
        -- Check if tokens_used column exists
        IF NOT EXISTS (
            SELECT 1 FROM information_schema.columns 
            WHERE table_schema = 'chatbot' 
            AND table_name = table_name 
            AND column_name = 'tokens_used'
        ) THEN
            EXECUTE format('ALTER TABLE chatbot.%I ADD COLUMN tokens_used INTEGER DEFAULT 0', table_name);
            RAISE NOTICE 'Added tokens_used column to %', table_name;
        ELSE
            RAISE NOTICE 'tokens_used column already exists in %', table_name;
        END IF;
    END LOOP;
END $$;

-- 6. Display summary
SELECT 
    'Migration completed successfully!' as status,
    COUNT(*) as total_clients,
    SUM(CASE WHEN token_limit_per_month IS NOT NULL THEN 1 ELSE 0 END) as clients_with_limits
FROM chatbot.clients;

SELECT 
    'Conversation tables updated:' as status,
    COUNT(*) as total_tables
FROM pg_tables 
WHERE schemaname = 'chatbot' 
AND tablename LIKE 'conversation_%';
