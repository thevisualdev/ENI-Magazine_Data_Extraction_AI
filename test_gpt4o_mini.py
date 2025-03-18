#!/usr/bin/env python
"""
Test script to check if the gpt-4o-mini model works with our API calls.
"""
import os
import yaml
import json
import dotenv
from openai import OpenAI

# Load environment variables
dotenv.load_dotenv()

def load_config():
    """Load configuration from config.yaml file."""
    with open('config.yaml', 'r') as f:
        return yaml.safe_load(f)

def get_api_key():
    """Get the API key from environment or config."""
    # From environment variable
    api_key = os.environ.get('OPENAI_API_KEY', '')
    
    # From config file as fallback
    if not api_key:
        config = load_config()
        api_key = config['openai'].get('api_key', '')
    
    return api_key

def test_with_response_format():
    """Test with response_format parameter."""
    api_key = get_api_key()
    if not api_key:
        print("Error: No API key found")
        return
    
    print(f"API key found: {'*' * (len(api_key) - 8) + api_key[-8:]}")
    
    config = load_config()
    model = config['openai']['model']
    print(f"Testing model: {model}")
    
    client = OpenAI(api_key=api_key)
    
    try:
        print("Attempting API call with response_format parameter...")
        response = client.chat.completions.create(
            model=model,
            temperature=0.3,
            max_tokens=500,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Return a simple JSON object with keys 'hello' and 'world'."}
            ],
            response_format={"type": "json_object"}
        )
        
        print(f"Response with response_format: {response.choices[0].message.content}")
        return True
    except Exception as e:
        print(f"Error with response_format: {str(e)}")
        return False

def test_without_response_format():
    """Test without response_format parameter."""
    api_key = get_api_key()
    if not api_key:
        print("Error: No API key found")
        return
    
    config = load_config()
    model = config['openai']['model']
    
    client = OpenAI(api_key=api_key)
    
    try:
        print("Attempting API call without response_format parameter...")
        response = client.chat.completions.create(
            model=model,
            temperature=0.3,
            max_tokens=500,
            messages=[
                {"role": "system", "content": "You are a helpful assistant. Respond with a JSON object only."},
                {"role": "user", "content": "Return a simple JSON object with keys 'hello' and 'world'."}
            ]
        )
        
        print(f"Response without response_format: {response.choices[0].message.content}")
        return True
    except Exception as e:
        print(f"Error without response_format: {str(e)}")
        return False

if __name__ == "__main__":
    print("Testing gpt-4o-mini compatibility...")
    
    # Test with response_format parameter
    with_format = test_with_response_format()
    
    # Test without response_format parameter
    without_format = test_without_response_format()
    
    # Print conclusion
    if with_format:
        print("\n✅ API calls WITH response_format parameter are working")
    else:
        print("\n❌ API calls WITH response_format parameter are failing")
    
    if without_format:
        print("✅ API calls WITHOUT response_format parameter are working")
    else:
        print("❌ API calls WITHOUT response_format parameter are failing")
    
    # Recommendation
    if not with_format and without_format:
        print("\nRECOMMENDATION: Update the code to NOT use response_format with this model")
    elif with_format and not without_format:
        print("\nRECOMMENDATION: Update the code to ALWAYS use response_format with this model")
    elif not with_format and not without_format:
        print("\nRECOMMENDATION: Check API key and model availability") 