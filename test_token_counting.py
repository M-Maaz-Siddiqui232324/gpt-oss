"""Test script for token counting functionality"""
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from utils import count_tokens, approximate_token_count

def test_token_counting():
    """Test token counting with various inputs"""
    
    test_cases = [
        ("Hello, how are you?", "Short greeting"),
        ("What is the process for submitting a leave request in FlowHCM?", "Medium question"),
        ("Can you explain in detail how the attendance module works, including clock-in, clock-out, overtime tracking, and shift management features?", "Long question"),
        ("", "Empty string"),
        ("A" * 1000, "Very long text (1000 chars)"),
    ]
    
    print("="*80)
    print("TOKEN COUNTING TEST")
    print("="*80)
    print()
    
    for text, description in test_cases:
        print(f"Test: {description}")
        print(f"Text: {text[:100]}{'...' if len(text) > 100 else ''}")
        print(f"Length: {len(text)} characters, {len(text.split())} words")
        
        # Count tokens
        tokens = count_tokens(text)
        approx_tokens = approximate_token_count(text)
        
        print(f"Tokens (tiktoken): {tokens}")
        print(f"Tokens (approximate): {approx_tokens}")
        print(f"Difference: {abs(tokens - approx_tokens)}")
        print("-"*80)
        print()
    
    # Test conversation scenario
    print("="*80)
    print("CONVERSATION SCENARIO TEST")
    print("="*80)
    print()
    
    user_query = "How do I submit a leave request?"
    bot_response = """To submit a leave request in FlowHCM:

1. Navigate to the Leave Module
2. Click on "Apply Leave"
3. Select the leave type (Annual, Sick, etc.)
4. Choose the start and end dates
5. Enter a reason for your leave
6. Submit the request

Your manager will be notified and can approve or reject the request. You'll receive a notification once it's processed."""
    
    user_tokens = count_tokens(user_query)
    bot_tokens = count_tokens(bot_response)
    total_tokens = user_tokens + bot_tokens
    
    print(f"User Query: {user_query}")
    print(f"User Tokens: {user_tokens}")
    print()
    print(f"Bot Response: {bot_response[:100]}...")
    print(f"Bot Tokens: {bot_tokens}")
    print()
    print(f"Total Tokens: {total_tokens}")
    print()
    
    # Estimate cost (example: $0.002 per 1K tokens)
    cost_per_1k = 0.002
    estimated_cost = (total_tokens / 1000) * cost_per_1k
    print(f"Estimated Cost (at ${cost_per_1k}/1K tokens): ${estimated_cost:.6f}")
    print()
    
    # Monthly usage projection
    queries_per_day = 100
    days_per_month = 30
    monthly_tokens = total_tokens * queries_per_day * days_per_month
    monthly_cost = (monthly_tokens / 1000) * cost_per_1k
    
    print(f"Monthly Projection ({queries_per_day} queries/day):")
    print(f"  Total Tokens: {monthly_tokens:,}")
    print(f"  Estimated Cost: ${monthly_cost:.2f}")
    print()
    
    print("="*80)
    print("TEST COMPLETED")
    print("="*80)


if __name__ == "__main__":
    test_token_counting()
