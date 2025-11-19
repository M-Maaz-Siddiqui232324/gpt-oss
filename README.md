# GPT-OSS RAG Chatbot with Multi-User Session Management

A production-ready RAG (Retrieval-Augmented Generation) chatbot with Redis-based session management supporting 100+ concurrent users.

## Features

✅ **Multi-User Sessions** - Isolated conversations for each user  
✅ **Session Archiving** - Automatic conversation history storage  
✅ **RAG System** - Context-aware responses using your documents  
✅ **FastAPI Backend** - RESTful API with automatic documentation  
✅ **Streamlit UI** - Interactive web interface  
✅ **Redis Integration** - Fast session management  
✅ **Local LLM** - Uses Ollama with GPT-OSS-20B model  

## Architecture

```
User → FastAPI → SessionManager → Redis (active sessions)
                      ↓
                 RAG System → LLM (Ollama)
                      ↓
              session_archives/ (permanent storage)
```

## Quick Start

### 1. Prerequisites

```bash
# Install Python dependencies
pip install -r requirements.txt

# Install Ollama
# Windows/Mac: https://ollama.com/download
# Linux: curl -fsSL https://ollama.com/install.sh | sh

# Pull the model
ollama pull gpt-oss:20b

# Install Redis
# Docker (easiest): docker run -d -p 6379:6379 redis
# Windows: https://github.com/microsoftarchive/redis/releases
# Linux: sudo apt install redis-server
# Mac: brew install redis
```

### 2. Setup

```bash
# Clone repository
git clone https://github.com/M-Maaz-Siddiqui232324/gpt-oss.git
cd gpt-oss

# Add your documents to docs/ folder
mkdir docs
# Add your .txt, .md, .docx files here

# Start Redis
redis-server  # or docker run -d -p 6379:6379 redis
```

### 3. Run

```bash
# Start API
python apps/api.py

# Or start Streamlit UI (in another terminal)
streamlit run apps/streamlit_app.py
```

## API Usage

### Start Conversation
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is this about?"}'
```

Response:
```json
{
  "session_id": "abc-123-...",
  "response": "Based on the documents...",
  "sources_count": 3
}
```

### Continue Conversation
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "abc-123-...",
    "query": "Tell me more"
  }'
```

### End Session (Archives Automatically)
```bash
curl -X POST http://localhost:8000/session/abc-123/end
```

### View Active Sessions
```bash
curl http://localhost:8000/sessions
```

### View Archived Sessions
```bash
curl http://localhost:8000/archives
```

## API Documentation

Interactive API docs available at: `http://localhost:8000/docs`

## Session Management

### How It Works

1. **User sends query** → API generates unique `session_id`
2. **Conversation stored** → Redis keeps active sessions
3. **Context maintained** → Each message uses conversation history
4. **Session ends** → Automatically archived to JSON file
5. **Multi-user support** → 100+ isolated sessions simultaneously

### Session Lifecycle

```
Create → Active (Redis) → End → Archive (JSON file)
```

### Archive Location

Sessions saved to: `session_archives/session_{id}_{timestamp}.json`

## Configuration

Edit `src/config.py`:

```python
# Redis
REDIS_HOST = "localhost"
REDIS_PORT = 6379
SESSION_EXPIRY_SECONDS = 1800  # 30 minutes

# Archive
ARCHIVE_FOLDER = "session_archives"

# Model
MODEL_NAME = "gpt-oss:20b"
OLLAMA_BASE_URL = "http://localhost:11434"

# RAG
TOP_K_RETRIEVAL = 10
TOP_K_CONTEXT = 7
```

## Project Structure

```
gpt-oss/
├── apps/
│   ├── api.py              # FastAPI server
│   └── streamlit_app.py    # Streamlit UI
├── src/
│   ├── config.py           # Configuration
│   ├── session_manager.py  # Redis session management
│   ├── rag_system.py       # RAG orchestration
│   ├── processing/         # Document processing
│   ├── retrieval/          # Vector search
│   └── generation/         # LLM integration
├── docs/                   # Your documents (add here)
├── session_archives/       # Archived conversations
├── data/                   # Vector store cache
└── requirements.txt
```

## Testing

```bash
# Test session management
python test_sessions.py

# Test archiving
python test_session_archive.py

# Test API examples
python example_api_usage.py
```

## Integration Example

### JavaScript
```javascript
// Start conversation
const response = await fetch('http://localhost:8000/query', {
  method: 'POST',
  headers: {'Content-Type': 'application/json'},
  body: JSON.stringify({query: 'Hello'})
});
const {session_id} = await response.json();

// Continue conversation
await fetch('http://localhost:8000/query', {
  method: 'POST',
  headers: {'Content-Type': 'application/json'},
  body: JSON.stringify({
    session_id: session_id,
    query: 'Tell me more'
  })
});

// End on logout
await fetch(`http://localhost:8000/session/${session_id}/end`, {
  method: 'POST'
});
```

### Python
```python
import requests

# Start
r = requests.post('http://localhost:8000/query', 
  json={'query': 'Hello'})
session_id = r.json()['session_id']

# Continue
requests.post('http://localhost:8000/query',
  json={'session_id': session_id, 'query': 'More info'})

# End
requests.post(f'http://localhost:8000/session/{session_id}/end')
```

## Documentation

- **Quick Start**: `QUICKSTART_SESSIONS.md`
- **Redis Setup**: `REDIS_SETUP.md`
- **Session Archives**: `SESSION_ARCHIVE_GUIDE.md`
- **API Reference**: `QUICK_REFERENCE.md`
- **Changes**: `CHANGES_SUMMARY.md`

## Requirements

- Python 3.8+
- Redis 5.0+
- Ollama with GPT-OSS-20B model
- 8GB+ RAM recommended

## Deployment

### Production Checklist

1. Use managed Redis (AWS ElastiCache, Redis Cloud)
2. Update `REDIS_HOST` in config
3. Set up HTTPS/SSL
4. Configure CORS properly
5. Add authentication
6. Monitor with `/health` endpoint
7. Set up log aggregation
8. Configure Redis persistence

## Troubleshooting

**Redis not connecting?**
```bash
redis-cli ping  # Should return PONG
```

**Ollama not responding?**
```bash
ollama list  # Check if model is installed
ollama pull gpt-oss:20b
```

**No documents loaded?**
- Add files to `docs/` folder
- Supported: .txt, .md, .docx

**Session not found?**
- Session expired (30 min)
- Start new conversation without session_id

## Performance

- **Concurrent Users**: 100+
- **Session Storage**: Redis (in-memory, <1ms)
- **Response Time**: 2-5 seconds (depends on LLM)
- **Scalability**: Horizontal (multiple API servers + shared Redis)

## License

MIT License - See LICENSE file

## Contributing

1. Fork the repository
2. Create feature branch
3. Commit changes
4. Push to branch
5. Open Pull Request

## Support

- **Issues**: https://github.com/M-Maaz-Siddiqui232324/gpt-oss/issues
- **Discussions**: https://github.com/M-Maaz-Siddiqui232324/gpt-oss/discussions

## Acknowledgments

- Ollama for local LLM support
- Redis for session management
- FastAPI for API framework
- Streamlit for UI

---

**Built with ❤️ for production-ready RAG applications**
