import json
import yaml
import os
from typing import Dict, Any, Optional
from openai import OpenAI
import dotenv

# Load environment variables from .env file
dotenv.load_dotenv()

def load_config() -> Dict[str, Any]:
    """Load configuration from config.yaml file."""
    with open('config.yaml', 'r') as f:
        return yaml.safe_load(f)

def create_openai_client(api_key: str) -> OpenAI:
    """
    Create an OpenAI client with the provided API key.
    
    Args:
        api_key: The OpenAI API key
        
    Returns:
        OpenAI client instance
    """
    return OpenAI(api_key=api_key)

def setup_openai_api(config: Dict[str, Any]) -> str:
    """
    Set up the OpenAI API by retrieving the API key from environment 
    variables or configuration file.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        API key (str)
    """
    # First try to get API key from environment variable
    api_key = os.environ.get('OPENAI_API_KEY', '')
    
    # If not found in environment, try to get from session state (if we're in Streamlit)
    try:
        import streamlit as st
        if not api_key and 'api_key' in st.session_state:
            api_key = st.session_state.api_key
    except ImportError:
        # Not running in Streamlit, so session state is not available
        pass
    
    # As a fallback, try to get from config dictionary
    if not api_key and config and 'openai' in config and 'api_key' in config['openai']:
        api_key = config['openai'].get('api_key', '')
    
    if not api_key:
        raise ValueError("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable or provide it in the config.yaml file.")
    
    return api_key

def clean_json_content(content: str) -> str:
    # Clean the JSON content - remove markdown code blocks if present
    if '```' in content:
        # Try to extract between markdown code blocks
        try:
            start_block = content.find('```')
            language_indicator = content.find('\n', start_block)
            start_json = content.find('\n', start_block) + 1
            end_json = content.rfind('```')
            
            # Extract the JSON content between the backticks
            if start_json > 0 and end_json > start_json:
                content = content[start_json:end_json].strip()
        except:
            pass
        
        # Fallback: remove all ``` markers
        content = content.replace('```json', '').replace('```', '').strip()
    
    # Make sure the content starts with { and ends with }
    content = content.strip()
    if not content.startswith('{'):
        first_brace = content.find('{')
        if first_brace >= 0:
            content = content[first_brace:]
        else:
            content = '{' + content
            
    if not content.endswith('}'):
        last_brace = content.rfind('}')
        if last_brace >= 0:
            content = content[:last_brace+1]
        else:
            content = content + '}'
    
    import re
    
    # Use a much simpler approach - just try to parse the JSON directly first
    try:
        # First, try parsing as is
        data = json.loads(content)
        return json.dumps(data)
    except Exception as parse_error:
        print(f"Initial parsing failed: {parse_error}, trying basic fixes")
        
        # If that fails, try some basic fixes
        try:
            # Replace single quotes used for field names with double quotes
            content = re.sub(r"([{,]\s*)'([^']+)'(\s*:)", r'\1"\2"\3', content)
            
            # Fix field names without quotes
            content = re.sub(r'([{,]\s*)([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1"\2":', content)
            
            # Fix trailing commas
            content = content.replace(',}', '}')
            content = content.replace(',\n}', '\n}')
            
            # Try parsing again
            data = json.loads(content)
            return json.dumps(data)
        except Exception as fix_error:
            print(f"Fix attempt failed: {fix_error}, trying field extraction")
            
            # If that still fails, try to extract individual fields
            fallback = {
                "magazine": "",
                "magazine_no": "",
                "author": "",
                "title": "",
                "abstract": "Could not parse full JSON response",
                "theme": "",
                "format": "",
                "geographic_area": "",
                "keywords": ""
            }
            
            # Try to extract field values with regex
            fields = ["magazine", "magazine_no", "author", "title", "abstract", 
                       "theme", "format", "geographic_area", "keywords"]
            
            for field in fields:
                # Try to extract the field with a simple pattern
                # This handles both "field": "value" and 'field': 'value' formats
                pattern = r'["\']' + field + r'["\']\s*:\s*["\']([^"\']*)["\']'
                match = re.search(pattern, content)
                if match:
                    fallback[field] = match.group(1)
            
            print(f"Created fallback JSON with {sum(1 for v in fallback.values() if v)} fields")
            return json.dumps(fallback)

def reconcile_metadata(file_data: Dict[str, Any], config: Dict[str, Any] = None) -> Dict[str, str]:
    """
    Reconcile metadata using OpenAI API.
    
    Args:
        file_data: File data with extracted fields
        config: Configuration dictionary
        
    Returns:
        Dict containing reconciled fields
    """
    try:
        # Make sure we have a valid config object
        if config is None:
            # Load config if not provided
            try:
                config = load_config()
                print("Loaded config from file as none was provided for reconciliation")
            except Exception as config_error:
                print(f"Error loading config for reconciliation: {config_error}")
                # Create a minimal default config
                config = {
                    'openai': {
                        'model': 'gpt-3.5-turbo',
                        'temperature': 0.1,
                        'max_tokens': 1000
                    },
                    'prompts': {
                        'reconcile_metadata': "Reconcile the following metadata from file: {file_path}\nMagazine: {magazine}\nMagazine Number: {magazine_no}\nAuthor: {author}\nTitle: {title}\n\nText content: {text_content}\n\nReturn a JSON with corrected fields."
                    }
                }
                print("Created default config as fallback for reconciliation")
            
        # First check the config for the API key (that should be set by the batch_processor)
        api_key = None
        
        # Log all possible sources for the API key
        if 'openai' in config and 'api_key' in config['openai'] and config['openai']['api_key']:
            api_key = config['openai']['api_key']
            print(f"Found API key in config object for reconciliation")
        else:
            print(f"API key not found in config object for reconciliation")
            
        # If not in config, try environment 
        if not api_key:
            api_key = os.environ.get('OPENAI_API_KEY')
            if api_key:
                print(f"Found API key in environment for reconciliation")
            else:
                print(f"API key not found in environment for reconciliation")
        
        if not api_key:
            raise ValueError("OpenAI API key not found for metadata reconciliation.")
        
        # Create OpenAI client with the API key
        client = OpenAI(api_key=api_key)
        
        print(f"Starting metadata reconciliation for file: {file_data.get('title', 'Unknown')}")
        print(f"API key retrieved successfully")
        
        # Get OpenAI settings from config
        model = config['openai'].get('model', "gpt-3.5-turbo")
        temperature = config['openai'].get('temperature', 0.1)
        max_tokens = config['openai'].get('max_tokens', 1000)
        
        print(f"Using model: {model}")
        
        # Get the prompt template
        if 'prompts' in config and 'reconcile_metadata' in config['prompts']:
            prompt_template = config['prompts']['reconcile_metadata']
        else:
            # Fallback prompt template if not in config
            prompt_template = "Reconcile the following metadata from file: {file_path}\nMagazine: {magazine}\nMagazine Number: {magazine_no}\nAuthor: {author}\nTitle: {title}\n\nText content: {text_content}\n\nReturn a JSON with corrected fields."
            print("Using fallback prompt template for reconciliation as it wasn't found in config")
        
        # Prepare the prompt with file data - using a safer approach
        # Instead of using string formatting which can fail if there are unexpected placeholders,
        # build the prompt directly
        
        prompt = "Reconcile the following metadata from file:\n"
        prompt += f"Magazine: {file_data.get('magazine', '')}\n"
        prompt += f"Magazine Number: {file_data.get('magazine_no', '')}\n"
        prompt += f"Author: {file_data.get('author', '')}\n"
        prompt += f"Title: {file_data.get('title', '')}\n\n"
        prompt += f"Text content: {file_data.get('text_content', '')}\n\n"
        prompt += "Return a JSON with the following fields: magazine, magazine_no, author, title, abstract, theme, format, geographic_area, keywords."
        
        # Check if response_format is supported for this model
        response_format_models = ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo", "gpt-4o", "gpt-4-turbo-preview", "gpt-4o-mini"]
        model_base = model.split('-')[0] + '-' + model.split('-')[1]  # Get base model name
        
        # Make the API call based on model support
        if model in response_format_models or model_base in response_format_models:
            print("Making API call with response_format=json")
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that reconciles metadata. Return valid JSON with fields: magazine, magazine_no, author, title, abstract, theme, format, geographic_area, keywords."},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                response_format={"type": "json_object"}
            )
        else:
            print("Making API call without using response_format parameter")
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that reconciles metadata. Return only valid JSON with fields: magazine, magazine_no, author, title, abstract, theme, format, geographic_area, keywords."},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                max_tokens=max_tokens
            )
            
        print("Successfully received response from OpenAI API")
        
        # Get the content from the response
        content = response.choices[0].message.content
        
        print("Got JSON content from response")
        
        # Clean and extract JSON content
        json_content = clean_json_content(content)
            
        print(f"Cleaned JSON content: {json_content[:100]}...")
        
        try:
            # Parse the JSON
            data = json.loads(json_content)
            
            print("Successfully parsed JSON response")
            
            # Ensure all required fields are present
            required_fields = ['magazine', 'magazine_no', 'author', 'title', 'abstract', 'theme', 'format', 'geographic_area', 'keywords']
            for field in required_fields:
                if field not in data:
                    print(f"Missing field '{field}' in response, adding default value from original data")
                    if field in file_data:
                        data[field] = file_data.get(field, '')
                    elif field == 'abstract':
                        data[field] = "No abstract available"
                    elif field == 'keywords':
                        data[field] = "unknown, missing, unspecified"
                    else:
                        data[field] = "Unknown"
            
            # Create a new dictionary with the validated/corrected data
            reconciled_data = {
                'abstract': data.get('abstract', file_data.get('abstract', 'Error extracting abstract')),
                'theme': data.get('theme', file_data.get('theme', 'Unknown')),
                'format': data.get('format', file_data.get('format', 'Unknown')),
                'geographic_area': data.get('geographic_area', file_data.get('geographic_area', 'Unknown')),
                'keywords': data.get('keywords', file_data.get('keywords', 'error, extraction, failed')),
                'magazine': data.get('magazine', file_data.get('magazine', '')),
                'magazine_no': data.get('magazine_no', file_data.get('magazine_no', '')),
                'author': data.get('author', file_data.get('author', '')),
                'title': data.get('title', file_data.get('title', '')),
            }
            
            # Keep the original data for reference
            reconciled_data['original_magazine'] = file_data.get('magazine', '')
            reconciled_data['original_magazine_no'] = file_data.get('magazine_no', '')
            reconciled_data['original_author'] = file_data.get('author', '')
            reconciled_data['original_title'] = file_data.get('title', '')
            
            # Preserve other important fields from the original data
            for key in ['full_path', 'text_content', 'preview_image']:
                if key in file_data:
                    reconciled_data[key] = file_data[key]
            
            return reconciled_data
            
        except json.JSONDecodeError as json_error:
            print(f"JSON parsing error in reconciliation: {json_error}")
            print(f"Problematic JSON: {json_content}")
            
            # Return the original file data but mark that reconciliation failed
            result = file_data.copy()
            result['abstract'] = f"Error parsing JSON during reconciliation: {str(json_error)}"
            
            # Keep the original data for reference if not already present
            if 'original_magazine' not in result:
                result['original_magazine'] = file_data.get('magazine', '')
            if 'original_magazine_no' not in result:
                result['original_magazine_no'] = file_data.get('magazine_no', '')
            if 'original_author' not in result:
                result['original_author'] = file_data.get('author', '')
            if 'original_title' not in result:
                result['original_title'] = file_data.get('title', '')
                
            return result
        
    except Exception as e:
        print(f"Error reconciling metadata: {e}")
        # Return error response - preserve original data 
        return {
            'abstract': file_data.get('abstract', f"Error during reconciliation: {str(e)}"),
            'theme': file_data.get('theme', 'error'),
            'format': file_data.get('format', 'error'),
            'geographic_area': file_data.get('geographic_area', 'error'),
            'keywords': file_data.get('keywords', 'error, reconciliation, failed'),
            'magazine': file_data.get('magazine', ''),
            'magazine_no': file_data.get('magazine_no', ''),
            'author': file_data.get('author', ''),
            'title': file_data.get('title', '')
        }

def batch_reconcile_metadata(file_data_list: list, config: Optional[Dict[str, Any]] = None) -> list:
    """
    Process a batch of files for metadata reconciliation.
    
    Args:
        file_data_list: List of file data dictionaries from the first extraction phase
        config: Optional configuration dictionary
        
    Returns:
        List of dictionaries with validated and corrected metadata
    """
    if not config:
        config = load_config()
    
    results = []
    
    for file_data in file_data_list:
        # Skip files that don't have text content
        if 'text_content' not in file_data or not file_data['text_content']:
            results.append(file_data)
            continue
        
        # Reconcile metadata
        reconciled_data = reconcile_metadata(file_data, config)
        results.append(reconciled_data)
    
    return results 