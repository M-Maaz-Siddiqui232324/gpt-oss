
CREATE SCHEMA IF NOT EXISTS chatbot;

SET search_path TO chatbot, public;

CREATE TABLE IF NOT EXISTS chatbot.clients (
    client_id SERIAL PRIMARY KEY,
    client_name VARCHAR(255) NOT NULL,
    api_key VARCHAR(255) UNIQUE NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    is_active BOOLEAN DEFAULT TRUE
);

CREATE TABLE IF NOT EXISTS chatbot.sessions (
    session_id UUID PRIMARY KEY,
    client_id INTEGER NOT NULL,
    messages JSONB NOT NULL DEFAULT '[]'::jsonb,
    session_start_time TIMESTAMP NOT NULL,
    session_end_time TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (client_id) REFERENCES chatbot.clients(client_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_sessions_client_id ON chatbot.sessions(client_id);
CREATE INDEX IF NOT EXISTS idx_sessions_start_time ON chatbot.sessions(session_start_time);
CREATE INDEX IF NOT EXISTS idx_sessions_end_time ON chatbot.sessions(session_end_time);
CREATE INDEX IF NOT EXISTS idx_clients_api_key ON chatbot.clients(api_key);

-- Insert a default client for testing
INSERT INTO chatbot.clients (client_name, api_key) 
VALUES ('FlowHCM', 'flowhcm_default_key_12345')
ON CONFLICT (api_key) DO NOTHING;
