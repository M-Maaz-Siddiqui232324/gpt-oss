# FlowHCM Chatbot - Complete Architecture & Implementation Documentation

## Executive Summary

The FlowHCM Chatbot is an **AI-powered conversational assistant** built using **Retrieval-Augmented Generation (RAG)** architecture. It provides intelligent responses to user queries about FlowHCM HR management software by leveraging local documentation and a locally-hosted Large Language Model (LLM).

**Key Technologies:**
- **LLM**: GPT-OSS-20B (20 billion parameters) via Ollama
- **Vector Database**: FAISS (Facebook AI Similarity Search)
- **Embeddings**: BAAI/bge-base-en-v1.5 for semantic search
- **Backend API**: FastAPI (Python)
- **Frontend UI**: Streamlit
- **Database**: PostgreSQL for session persistence
- **Document Processing**: python-docx for Word documents

**Core Capabilities:**
- Semantic document search across FlowHCM documentation
- Context-aware conversational responses
- Session management with conversation history
- Multi-turn dialogue support
- Real-time response generation
- Session persistence to PostgreSQL database

---

## System Architecture Overview

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     FlowHCM Chatbot System                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐         ┌──────────────┐                    │
│  │  Streamlit   │────────▶│   FastAPI    │                    │
│  │     UI       │◀────────│   Backend    │                    │
│  │  (Port 8501) │  HTTP   │  (Port 8000) │                    │
│  └──────────────┘         └───────┬──────┘                    │
│                                   │                             │
│                          ┌────────┴────────┐                   │
│                          │                 │                   │
│                          ▼                 ▼                   │
│                   ┌─────────────┐   ┌─────────────┐           │
│                   │  RAG System │   │  PostgreSQL │           │
│                   │             │   │  Database   │           │
│                   └──────┬──────┘   └─────────────┘           │
│                          │                                     │
│              ┌───────────┼───────────┐                        │
│              │           │           │                        │
│              ▼           ▼           ▼                        │
│         ┌────────┐  ┌────────┐  ┌────────┐                  │
│         │ Vector │  │  LLM   │  │  Docs  │                  │
│         │ Store  │  │ Engine │  │Processor│                 │
│         │(FAISS) │  │(Ollama)│  │        │                  │
│         └────────┘  └────────┘  └────────┘                  │
│                                                               │
└───────────────────────────────────────────────────────────────┘
```


### RAG (Retrieval-Augmented Generation) Pipeline

```
User Query
    ↓
┌─────────────────────────────────────────────────────────────┐
│ 1. QUERY PROCESSING                                         │
│    - Receive user input                                     │
│    - Extract session context                                │
└────────────────────────┬────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────┐
│ 2. SEMANTIC RETRIEVAL                                       │
│    - Encode query to vector embedding                       │
│    - Search FAISS index for similar chunks                  │
│    - Retrieve top-K relevant document chunks                │
│    - Apply relevance threshold filtering                    │
└────────────────────────┬────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────┐
│ 3. CONTEXT ASSEMBLY                                         │
│    - Combine retrieved document chunks                      │
│    - Add conversation history (recent exchanges)            │
│    - Build structured prompt                                │
└────────────────────────┬────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────┐
│ 4. LLM GENERATION                                           │
│    - Send prompt to Ollama (GPT-OSS-20B)                   │
│    - Generate contextual response                           │
│    - Apply stop sequences and cleanup                       │
└────────────────────────┬────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────┐
│ 5. RESPONSE POST-PROCESSING                                 │
│    - Clean up artifacts and incomplete sentences            │
│    - Store in session history                               │
│    - Return to user                                         │
└─────────────────────────────────────────────────────────────┘
```

---

## Repository Structure

```
chatbot/
├── apps/                           # Application entry points
│   ├── api.py                      # FastAPI backend server
│   ├── streamlit_app.py            # Streamlit UI application
│   └── __init__.py
│
├── src/                            # Core source code
│   ├── config.py                   # Configuration settings
│   ├── rag_system.py               # Main RAG orchestrator
│   ├── utils.py                    # Utility functions
│   ├── fastapi_session_manager.py  # Session management
│   │
│   ├── database/                   # Database layer
│   │   ├── postgres_manager.py     # PostgreSQL operations
│   │   ├── schema.sql              # Database schema
│   │   └── __init__.py
│   │
│   ├── processing/                 # Document processing
│   │   ├── document_processor.py   # Document loading
│   │   ├── chunking.py             # Semantic chunking
│   │   └── __init__.py
│   │
│   ├── retrieval/                  # Vector search
│   │   ├── vector_store.py         # FAISS index management
│   │   ├── retriever.py            # Semantic retrieval
│   │   └── __init__.py
│   │
│   └── generation/                 # LLM integration
│       ├── llm_engine.py           # Ollama interface
│       ├── prompts.py              # Prompt templates
│       └── __init__.py
│
├── docs/                           # Documentation corpus
│   ├── Employee Module (5.10).docx
│   ├── Attendance Module (5.10).docx
│   ├── Leave Module (5.10).docx
│   ├── Payroll Module (5.10).docx
│   └── ... (18 total .docx files)
│
├── data/                           # Persistent data
│   ├── faiss_index.bin             # FAISS vector index
│   └── document_chunks.pkl         # Serialized chunks
│
├── myapp files/                    # Frontend integration files
│   ├── chatbot.html                # HTML widget
│   ├── chatbotcontroller.js        # AngularJS controller
│   ├── chatbotservice.js           # AngularJS service
│   └── chatbotcontroller.cs        # C# backend controller
│
├── .env                            # Environment variables
├── requirements.txt                # Python dependencies
└── architecture.md                 # This document
```

---

