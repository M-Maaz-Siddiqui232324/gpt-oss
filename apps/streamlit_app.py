"""Streamlit UI for the RAG chatbot"""
import streamlit as st
import streamlit.components.v1 as components
import logging
import os
import sys
import requests
from typing import List, Dict, Optional

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from config import LOG_LEVEL, API_PORT, DEFAULT_MAX_TOKENS, DEFAULT_TEMPERATURE, DEFAULT_TOP_P
import utils

# Setup logging
utils.setup_logging(LOG_LEVEL)
logger = logging.getLogger(__name__)

# FastAPI backend URL
# Use localhost instead of 0.0.0.0 for client connections
API_BASE_URL = f"http://localhost:{API_PORT}"

# Page config
st.set_page_config(
    page_title="FlowHCM GPT-OSS Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

logger.info("="*60)
logger.info("FlowHCM Chatbot Starting Up")
logger.info("="*60)


def check_api_health():
    """Check if FastAPI backend is running"""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            logger.info("FastAPI backend is healthy")
            return data
        else:
            logger.error(f"API health check failed: {response.status_code}")
            return None
    except requests.exceptions.RequestException as e:
        logger.error(f"Cannot connect to FastAPI backend: {e}")
        return None


def get_documents():
    """Get list of documents from API"""
    try:
        response = requests.get(f"{API_BASE_URL}/documents", timeout=5)
        if response.status_code == 200:
            return response.json()
        return None
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching documents: {e}")
        return None


def send_query(query: str, max_tokens: int, temperature: float, top_p: float, session_cookie: Optional[str] = None):
    """Send query to FastAPI backend"""
    try:
        headers = {}
        cookies = {}
        
        # Add session cookie if available
        if session_cookie:
            cookies["session"] = session_cookie
        
        payload = {
            "query": query,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p
        }
        
        response = requests.post(
            f"{API_BASE_URL}/query",
            json=payload,
            cookies=cookies,
            timeout=180  # 3 minutes for LLM generation
        )
        
        if response.status_code == 200:
            data = response.json()
            # Extract session cookie from response
            new_session_cookie = response.cookies.get("session")
            return data, new_session_cookie
        else:
            logger.error(f"Query failed: {response.status_code} - {response.text}")
            return None, None
            
    except requests.exceptions.RequestException as e:
        logger.error(f"Error sending query: {e}")
        return None, None








def main():
    logger.info("Starting Streamlit app")
    
    # Title
    st.title("🤖 FlowHCM Chatbot")
    st.markdown("**Powered by GPT-OSS-20B (Local) via FastAPI**")
    
    # Check API health
    health_data = check_api_health()
    if health_data is None:
        st.error("❌ Cannot connect to FastAPI backend!")
        st.info(f"Please ensure the FastAPI server is running at {API_BASE_URL}")
        st.code(f"python apps/api.py", language="bash")
        return
    
    # Initialize session cookie in Streamlit session state
    if "session_cookie" not in st.session_state:
        st.session_state.session_cookie = None
        logger.info("New Streamlit session - no API session yet")
    
    # Sidebar
    with st.sidebar:
        st.header("📋 Document Library")
        
        # Get documents from API
        docs_data = get_documents()
        if docs_data and docs_data.get("documents"):
            for doc in docs_data["documents"]:
                with st.expander(f"📄 {doc['name']}"):
                    st.write(f"**Type:** {doc['type']}")
                    st.write(f"**Size:** {doc['size']} chars")
        else:
            st.info("No documents loaded")
        
        st.markdown("---")
        
        # Settings
        st.header("⚙️ Settings")
        max_tokens = st.slider("Max Tokens", 100, 1000, DEFAULT_MAX_TOKENS, 50)
        temperature = st.slider("Temperature", 0.1, 1.0, DEFAULT_TEMPERATURE, 0.1)
        top_p = st.slider("Top P", 0.5, 1.0, DEFAULT_TOP_P, 0.05)
        
        st.markdown("---")
        
        # System info
        st.subheader("📊 System Status")
        st.success(f"✅ **API Status:** {health_data['status']}")
        st.info(f"**Documents:** {health_data['documents_loaded']}")
        st.info(f"**Chunks:** {health_data['chunks_created']}")
        st.info(f"**Model Loaded:** {'Yes' if health_data['model_loaded'] else 'No'}")
        
        # Session info
        if st.session_state.session_cookie:
            # Show truncated session cookie
            cookie_preview = st.session_state.session_cookie[:16] + "..." if len(st.session_state.session_cookie) > 16 else st.session_state.session_cookie
            st.info(f"**Session:** {cookie_preview}")
        else:
            st.info("**Session:** Not started")
        
        # Clear chat (local only - no backend endpoint needed)
        if st.button("🗑️ Clear Chat", use_container_width=True):
            logger.info("Clearing chat history (local only)")
            st.session_state.messages = []
            st.rerun()
        
        # End session button (local only - no backend endpoint needed)
        if st.button("🔚 End Session", use_container_width=True):
            logger.info("Ending session (local only)")
            st.session_state.messages = []
            st.session_state.session_cookie = None
            st.rerun()
    
    # Initialize messages
    if "messages" not in st.session_state:
        logger.info("Initializing session state")
        st.session_state.messages = []
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask about your documentation..."):
        logger.info(f"User input received: '{prompt}'")
        
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Analyzing documentation..."):
                logger.info("Starting response generation via API")
                
                # Send query to FastAPI backend
                result, new_session_cookie = send_query(
                    prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    session_cookie=st.session_state.session_cookie
                )
                
                if result:
                    response = result["response"]
                    session_id = result.get("session_id", "unknown")
                    
                    # Update session cookie
                    if new_session_cookie:
                        st.session_state.session_cookie = new_session_cookie
                        logger.info(f"Session cookie updated: {session_id}")
                    
                    logger.info(f"Response generated: {len(response)} chars")
                    st.markdown(response)
                    
                    # Add to session state
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response
                    })
                else:
                    error_msg = "❌ Failed to get response from API"
                    st.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_msg
                    })
                
                logger.info("Session state updated")


if __name__ == "__main__":
    logger.info("Application started")
    try:
        main()
    except Exception as e:
        logger.error(f"Application error: {e}", exc_info=True)
        raise
