
CREATE SCHEMA IF NOT EXISTS chatbot;

SET search_path TO chatbot, public;

-- Clients table (stores client authentication info)
CREATE TABLE IF NOT EXISTS chatbot.clients (
    client_id INTEGER PRIMARY KEY,
    company_pin VARCHAR(50) NOT NULL,
    api_key VARCHAR(255) UNIQUE NOT NULL,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Sessions table (stores session metadata)
CREATE TABLE IF NOT EXISTS chatbot.sessions (
    id SERIAL PRIMARY KEY,
    session_id UUID UNIQUE NOT NULL,
    username VARCHAR(255) NOT NULL,
    fk_client_id INTEGER NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_active TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (fk_client_id) REFERENCES chatbot.clients(client_id) ON DELETE CASCADE
);

-- Conversation table for current month (December 2025)
-- Note: New tables will be created automatically each month with format: conversation_mmm_yyyy
CREATE TABLE IF NOT EXISTS chatbot.conversation_dec_2025 (
    conversation_id SERIAL PRIMARY KEY,
    fk_session_id INTEGER NOT NULL,
    user_message TEXT NOT NULL,
    chatbot_response TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (fk_session_id) REFERENCES chatbot.sessions(id) ON DELETE CASCADE
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_clients_api_key ON chatbot.clients(api_key);
CREATE INDEX IF NOT EXISTS idx_clients_company_pin ON chatbot.clients(company_pin);
CREATE INDEX IF NOT EXISTS idx_sessions_client_id ON chatbot.sessions(fk_client_id);
CREATE INDEX IF NOT EXISTS idx_sessions_session_id ON chatbot.sessions(session_id);
CREATE INDEX IF NOT EXISTS idx_sessions_username ON chatbot.sessions(username);
CREATE INDEX IF NOT EXISTS idx_sessions_last_active ON chatbot.sessions(last_active);
CREATE INDEX IF NOT EXISTS idx_conversation_dec_2025_session_id ON chatbot.conversation_dec_2025(fk_session_id);
CREATE INDEX IF NOT EXISTS idx_conversation_dec_2025_created_at ON chatbot.conversation_dec_2025(created_at);

-- Insert a default client for testing (optional)
INSERT INTO chatbot.clients (client_id, company_pin, api_key, is_active) 
VALUES (1, '1032', 'FLOW-1-flowhcm_default_key_12345', TRUE)
ON CONFLICT (client_id) DO NOTHING;
