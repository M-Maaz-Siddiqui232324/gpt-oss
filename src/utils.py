"""Utility functions"""
import re
import logging

logger = logging.getLogger(__name__)


def clean_response(response: str) -> str:
    """Clean up generated response"""
    if not response:
        return "I'm here to help! What would you like to know about FlowHCM?"
    
    stop_patterns = [
        "\nHuman:", "\nUser:", "\nAssistant:", "\nAI:", 
        "Human:", "User:", "Assistant:", "AI:", "ASSISTANT RESPONSE",
        "\nUSER QUESTION", "\nANSWER:", "\nDOCUMENTATION:",
        "\nYOUR RESPONSE", "\nUSER QUESTIONS",
        "USER QUESTION (", "USER QUESTIONS (",
        "\n\nHow do", "\n\nWhat is", "\n\nCan I", "\n\nWhere can",
        "\n\nHow to", "\n\nWhat are", "\n\nCan you", "\n\nWhere do",
        "\n\nIs there", "\n\nAre there",
        "\nRemember,", "\n\nRemember,",
        "what if i", "what if you",  
    ]
    
    for pattern in stop_patterns:
        if pattern.lower() in response.lower():
            pos = response.lower().find(pattern.lower())
            if pos != -1:
                response = response[:pos]
    
    response = re.sub(r'\s+', ' ', response).strip()
    
    if response and response[-1] not in '.!?':
        sentences = response.split('.')
        if len(sentences) > 1:
            response = '.'.join(sentences[:-1]) + '.'
    
    return response


def setup_logging(log_level: str = "INFO"):
    """Setup logging configuration"""
    logging.basicConfig(
        level=getattr(logging, log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )
