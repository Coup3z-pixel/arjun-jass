#!/usr/bin/env python3

import os
from dotenv import load_dotenv
from helper.llm.LLM import LLM

load_dotenv()

def test_hedonic_format():
    # Test the same model that's failing
    model_name = "together:jaspertsh08_a03a/Mixtral-8x7B-v0.1-50e88a5a-d680bff3"
    llm = LLM(model_name)
    
    # Simple test prompt for hedonic game
    test_prompt = """You are an agent in a social group. The following people are your friends: Bob, Charlie.
    Your enemies are: Diana.

    There are currently these groups:
    - Marketing: Charlie
    - Research: Alice, Bob
    - Solo: Diana

    You have two choices:
    - Stay in your current group
    - Leave to be alone

    IMPORTANT: You must respond with EXACTLY one of these values:
    - 1 if you want to STAY in your current group
    - 2 if you want to LEAVE to be alone

    NO OTHER VALUES ARE ACCEPTABLE. Only 1 or 2.

    Provide your reasoning for your choice.
    """
    
    print(f"Testing LLM: {model_name}")
    print("Prompt:")
    print(test_prompt)
    print("\n" + "="*50 + "\n")
    
    try:
        # Reset model first
        llm.restart_model()
        
        # Call the LLM with the standard ask method
        value, reasoning = llm.ask(test_prompt)
        
        print(f"Response value: {value} (type: {type(value)})")
        print(f"Response reasoning: {reasoning[:200]}...")
        
        # Check if it's valid
        if value in [1, 2]:
            print("✅ Valid response!")
        else:
            print(f"❌ Invalid response: {value}")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_hedonic_format()
