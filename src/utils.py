"""Utility functions"""
import re
import logging

logger = logging.getLogger(__name__)

# Token counting
try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False
    logger.warning("tiktoken not installed. Token counting will use approximate method.")


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


def count_tokens(text: str, model: str = "gpt-3.5-turbo") -> int:
    """
    Count tokens in text using tiktoken
    
    Args:
        text: Text to count tokens for
        model: Model name for encoding (default: gpt-3.5-turbo)
        
    Returns:
        Number of tokens
    """
    if not text:
        return 0
    
    if TIKTOKEN_AVAILABLE:
        try:
            encoding = tiktoken.encoding_for_model(model)
            return len(encoding.encode(text))
        except Exception as e:
            logger.warning(f"Error counting tokens with tiktoken: {e}")
            # Fallback to approximate method
            return approximate_token_count(text)
    else:
        return approximate_token_count(text)


def approximate_token_count(text: str) -> int:
    """
    Approximate token count (fallback method)
    Rough estimate: 1 token ≈ 4 characters or 0.75 words
    
    Args:
        text: Text to count tokens for
        
    Returns:
        Approximate number of tokens
    """
    if not text:
        return 0
    
    # Use word count method (more accurate than character count)
    words = len(text.split())
    return int(words / 0.75)
