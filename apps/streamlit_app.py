"""Streamlit UI for the RAG chatbot"""
import streamlit as st
import logging
import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from config import *
from rag_system import RAGSystem
import utils

# Setup logging
utils.setup_logging(LOG_LEVEL)
logger = logging.getLogger(__name__)

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


@st.cache_resource
def load_system():
    """Load the RAG system"""
    logger.info("Loading RAG system")
    
    # Check directories
    if not os.path.exists(DOCS_FOLDER):
        logger.error(f"Docs directory not found: {DOCS_FOLDER}")
        st.error(f"❌ Docs directory '{DOCS_FOLDER}' not found!")
        st.info("Please create a 'docs' folder and add your documentation files.")
        return None
    
    try:
        with st.spinner("Initializing RAG system..."):
            rag_system = RAGSystem()
            success = rag_system.initialize()
            
            if success:
                logger.info("RAG system loaded successfully")
                st.success(f"✅ Loaded {len(rag_system.documents)} documents with {len(rag_system.chunks)} chunks")
                return rag_system
            else:
                logger.error("Failed to initialize RAG system")
                st.error("❌ Failed to initialize RAG system")
                return None
    
    except Exception as e:
        logger.error(f"Error loading system: {e}", exc_info=True)
        st.error(f"❌ Error: {e}")
        return None


def main():
    logger.info("Starting Streamlit app")
    
    # Title
    st.title("🤖 FlowHCM Chatbot")
    st.markdown("**Powered by GPT-OSS-20B (Local)**")
    
    # Load system
    rag_system = load_system()
    if rag_system is None:
        return
    
    # Sidebar
    with st.sidebar:
        st.header("📋 Document Library")
        
        # Show documents
        if rag_system.documents:
            for doc in rag_system.documents:
                with st.expander(f"📄 {doc['name']}"):
                    st.write(f"**Type:** {doc['type']}")
                    st.write(f"**Size:** {len(doc['content'])} chars")
                    preview = doc['content'][:200] + "..." if len(doc['content']) > 200 else doc['content']
                    st.code(preview, language=doc['type'])
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
        st.info(f"**Device:** {rag_system.llm_engine.device}")
        st.info(f"**Documents:** {len(rag_system.documents)}")
        st.info(f"**Chunks:** {len(rag_system.chunks)}")
        
        if rag_system.vector_store.index:
            st.success(f"✅ **Vector Search:** {rag_system.vector_store.index.ntotal} embeddings")
        else:
            st.warning("⚠️ **Vector Search:** Not available")
        
        # Clear chat
        if st.button("🗑️ Clear Chat", use_container_width=True):
            logger.info("Clearing chat history")
            st.session_state.messages = []
            rag_system.clear_conversation()
            st.rerun()
    
    # Initialize session state
    if "messages" not in st.session_state:
        logger.info("Initializing session state")
        st.session_state.messages = []
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Show sources if available
            if message["role"] == "assistant" and "sources" in message and message["sources"]:
                with st.expander(f"📚 Sources ({len(message['sources'])})"):
                    for i, source in enumerate(message["sources"], 1):
                        st.markdown(f"**{i}. {source.source_file}** (score: {source.relevance_score:.3f})")
                        st.markdown(f"**Chunk ID:** {source.chunk_id}")
                        st.markdown(f"**Length:** {len(source.content)} characters")
                        # Unique key using message index and source index
                        msg_idx = st.session_state.messages.index(message)
                        st.text_area(
                            f"Full Content - Source {i}", 
                            source.content, 
                            height=300, 
                            disabled=True,
                            key=f"history_source_{msg_idx}_{i}"
                        )
                        st.markdown("---")
    
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
                logger.info("Starting response generation")
                
                response, context_docs = rag_system.query(
                    prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p
                )
                
                logger.info(f"Response generated: {len(response)} chars, {len(context_docs)} sources")
                
                st.markdown(response)
                
                # Show sources
                if context_docs:
                    with st.expander(f"📚 Sources ({len(context_docs)})"):
                        for i, doc in enumerate(context_docs, 1):
                            st.markdown(f"**{i}. {doc.source_file}** (score: {doc.relevance_score:.3f})")
                            st.markdown(f"**Chunk ID:** {doc.chunk_id}")
                            st.markdown(f"**Length:** {len(doc.content)} characters")
                            st.text_area(f"Full Content - Source {i}", doc.content, height=300, disabled=True)
                            st.markdown("---")
        
        # Add to session state
        st.session_state.messages.append({
            "role": "assistant",
            "content": response,
            "sources": context_docs
        })
        logger.info("Session state updated")


if __name__ == "__main__":
    logger.info("Application started")
    try:
        main()
    except Exception as e:
        logger.error(f"Application error: {e}", exc_info=True)
        raise
