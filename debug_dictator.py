#!/usr/bin/env python3

import os
from dotenv import load_dotenv
from helper.llm.LLM import LLM
from helper.game.dictator_game import DictatorGameAnswerFormat

load_dotenv()

def test_dictator_format():
    # Test the same model that's failing
    model_name = "together:jaspertsh08_a03a/Mixtral-8x7B-v0.1-50e88a5a-d680bff3"
    llm = LLM(model_name)
    
    # Simple test prompt for dictator game
    test_prompt = """You have just completed a software development project with 1 teammate(s). 

The client has paid $100 for the completed project, and this payment has been given directly to you to distribute among the team.

You must decide how to split the $100 between yourself and your teammate. Respond with a JSON object containing:
- "reasoning": your explanation for the split
- "keep_percent": the percentage you want to keep (0-100)
- "donate_percent": the percentage you want to give to your teammate (0-100)

Example: {"reasoning": "I think we should split equally", "keep_percent": 50, "donate_percent": 50}
"""
    
    print(f"Testing LLM: {model_name}")
    print("Prompt:")
    print(test_prompt)
    print("\n" + "="*50 + "\n")
    
    try:
        # Reset model first
        llm.restart_model()
        
        # Call the LLM with custom format
        response = llm.ask_with_custom_format(test_prompt, DictatorGameAnswerFormat)
        
        print(f"Response: {response}")
        print(f"Type: {type(response)}")
        print(f"Reasoning: {response.reasoning}")
        print(f"Keep percent: {response.keep_percent}")
        print(f"Donate percent: {response.donate_percent}")
        
        print("✅ Success!")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_dictator_format()
