
CREATE SCHEMA IF NOT EXISTS chatbot;

SET search_path TO chatbot, public;

CREATE TABLE IF NOT EXISTS chatbot.clients (
    client_id SERIAL PRIMARY KEY,
    company_pin VARCHAR(50) NOT NULL UNIQUE,
    api_key VARCHAR(255) NOT NULL UNIQUE,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    token_limit_per_month INTEGER DEFAULT 100000
);

CREATE TABLE IF NOT EXISTS chatbot.sessions (
    id SERIAL PRIMARY KEY,
    session_id UUID UNIQUE NOT NULL,
    username VARCHAR(255) NOT NULL,
    fk_client_id INTEGER NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_active TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (fk_client_id) REFERENCES chatbot.clients(client_id) ON DELETE CASCADE
);

-- Token usage tracking per client per month
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


CREATE TABLE IF NOT EXISTS chatbot.conversation_dec_2025 (
    conversation_id SERIAL PRIMARY KEY,
    fk_session_id INTEGER NOT NULL,
    user_message TEXT NOT NULL,
    chatbot_response TEXT NOT NULL,
    tokens_used INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (fk_session_id) REFERENCES chatbot.sessions(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_clients_api_key ON chatbot.clients(api_key);
CREATE INDEX IF NOT EXISTS idx_clients_company_pin ON chatbot.clients(company_pin);
CREATE INDEX IF NOT EXISTS idx_sessions_client_id ON chatbot.sessions(fk_client_id);
CREATE INDEX IF NOT EXISTS idx_sessions_session_id ON chatbot.sessions(session_id);
CREATE INDEX IF NOT EXISTS idx_sessions_username ON chatbot.sessions(username);
CREATE INDEX IF NOT EXISTS idx_sessions_last_active ON chatbot.sessions(last_active);
CREATE INDEX IF NOT EXISTS idx_conversation_dec_2025_session_id ON chatbot.conversation_dec_2025(fk_session_id);
CREATE INDEX IF NOT EXISTS idx_conversation_dec_2025_created_at ON chatbot.conversation_dec_2025(created_at);
CREATE INDEX IF NOT EXISTS idx_tokens_client_month ON chatbot.tokens(fk_client_id, month_year);

-- Documents table for tracking synced HR policy documents
CREATE TABLE IF NOT EXISTS chatbot.documents (
    document_id SERIAL PRIMARY KEY,
    announcement_id INTEGER NOT NULL,
    fk_client_id INTEGER NOT NULL,
    file_name VARCHAR(255) NOT NULL,
    document_title VARCHAR(500),
    file_extension VARCHAR(10),
    chunk_count INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    synced_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (fk_client_id) REFERENCES chatbot.clients(client_id) ON DELETE CASCADE,
    UNIQUE(announcement_id, fk_client_id)
);

CREATE INDEX IF NOT EXISTS idx_documents_client_id ON chatbot.documents(fk_client_id);
CREATE INDEX IF NOT EXISTS idx_documents_announcement_id ON chatbot.documents(announcement_id);

