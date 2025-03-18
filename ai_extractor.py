import json
import yaml
import time
from typing import Dict, Any, Optional, List
import os
from dotenv import load_dotenv
import traceback

# Direct import from openai with the latest package version
from openai import OpenAI

# Imports for text extraction
import docx
from pathlib import Path

# Load environment variables from .env file if it exists
load_dotenv()

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
    Setup the OpenAI API with prioritized configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        The API key that was successfully configured
    """
    # Prioritize:
    # 1. Environment variable
    # 2. Config file
    api_key = os.environ.get('OPENAI_API_KEY') or config['openai'].get('api_key')
    
    if not api_key:
        raise ValueError("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable or provide it in the config.yaml file.")
    
    return api_key

def extract_json_from_response(content):
    """
    Extract and clean JSON from the OpenAI API response.
    
    Args:
        content: String content from OpenAI API response
        
    Returns:
        Cleaned JSON string ready for parsing
    """
    try:
        # Skip if empty
        if not content:
            return "{}"
            
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
            except Exception as e:
                print(f"Error extracting from code blocks: {e}")
            
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
        
        # Use a much simpler approach - just try to parse the JSON directly
        try:
            # First, try parsing as is
            return json.dumps(json.loads(content))
        except:
            # If that fails, make some simple fixes
            # Replace single quotes with double quotes (for field names)
            content = re.sub(r"(?<![\\])(')([\w]+)(?:['])", r'"\2"', content)
            
            # Fix field names without quotes
            content = re.sub(r'([{,]\s*)([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1"\2":', content)
            
            # Fix trailing commas
            content = content.replace(',}', '}')
            content = content.replace(',\n}', '\n}')
            
            # Try parsing
            try:
                parsed = json.loads(content)
                return json.dumps(parsed)
            except:
                # One last attempt - rebuild the entire JSON structure
                fallback = {
                    "magazine": "",
                    "magazine_no": "",
                    "author": "",
                    "title": "",
                    "abstract": f"Could not parse JSON response",
                    "theme": "",
                    "format": "",
                    "geographic_area": "",
                    "keywords": ""
                }
                
                # Try to extract field values with regex
                fields = ["magazine", "magazine_no", "author", "title", "abstract", 
                          "theme", "format", "geographic_area", "keywords"]
                
                for field in fields:
                    # Match patterns like "field": "value" with various quote combinations
                    patterns = [
                        r'"' + field + r'":\s*"([^"]*)"',  # "field": "value"
                        r'"' + field + r'":\s*\'([^\']*)\'',  # "field": 'value'
                        r'\'' + field + r'\':\s*"([^"]*)"',  # 'field': "value"
                        r'\'' + field + r'\':\s*\'([^\']*)\'',  # 'field': 'value'
                    ]
                    
                    # Try each pattern
                    for pattern in patterns:
                        match = re.search(pattern, content)
                        if match:
                            fallback[field] = match.group(1)
                            break
                
                return json.dumps(fallback)
                
    except Exception as e:
        print(f"Error cleaning JSON: {e}")
        # Return a fallback JSON
        return json.dumps({
            "magazine": "",
            "magazine_no": "",
            "author": "",
            "title": "",
            "abstract": f"Error: {str(e)}",
            "theme": "",
            "format": "",
            "geographic_area": "",
            "keywords": ""
        })

def extract_fields_from_text(file_path, text_content, config=None):
    """
    Extract fields from text content using OpenAI API.
    
    Args:
        file_path: Path to the file
        text_content: Text content to extract fields from
        config: Configuration dictionary
        
    Returns:
        Dict containing extracted fields
    """
    try:
        # First check the config for the API key (that should be set by the batch_processor)
        api_key = None
        
        # Make sure we have a valid config object
        if config is None:
            # Load config if not provided
            try:
                config = load_config()
                print("Loaded config from file as none was provided")
            except Exception as config_error:
                print(f"Error loading config: {config_error}")
                # Create a minimal default config
                config = {
                    'openai': {
                        'model': 'gpt-3.5-turbo',
                        'temperature': 0.1,
                        'max_tokens': 1000
                    },
                    'prompts': {
                        'extract_fields': "Extract metadata from the following text. Return a JSON object with fields: magazine, magazine_no, author, title, abstract, theme, format, geographic_area, keywords.\n\n{text_content}"
                    }
                }
                print("Created default config as fallback")
        
        # Log all possible sources for the API key
        if config and 'openai' in config and 'api_key' in config['openai'] and config['openai']['api_key']:
            api_key = config['openai']['api_key']
            print(f"Found API key in config object")
        else:
            print(f"API key not found in config object")
            
        # If not in config, try environment 
        if not api_key:
            api_key = os.environ.get('OPENAI_API_KEY')
            if api_key:
                print(f"Found API key in environment")
            else:
                print(f"API key not found in environment")
        
        if not api_key:
            raise ValueError("OpenAI API key not found for extraction.")
        
        # Create OpenAI client with the API key
        client = OpenAI(api_key=api_key)
        
        # Get the filename from the path
        filename = os.path.basename(file_path)
        print(f"Starting extraction for file: {filename}")
        print(f"API key retrieved successfully")
        
        # Get OpenAI settings from config
        model = config['openai'].get('model', "gpt-3.5-turbo")
        temperature = config['openai'].get('temperature', 0.1)
        max_tokens = config['openai'].get('max_tokens', 1000)
        
        print(f"Using model: {model}")
        
        # Get the prompt template
        if 'prompts' in config and 'extract_fields' in config['prompts']:
            prompt_template = config['prompts']['extract_fields']
        else:
            # Fallback prompt template if not in config
            prompt_template = "Extract metadata from the following text. Return a JSON object with fields: magazine, magazine_no, author, title, abstract, theme, format, geographic_area, keywords.\n\n{text_content}"
            print("Using fallback prompt template as it wasn't found in config")
        
        # Prepare the prompt - using a safer approach that doesn't rely on string formatting
        # which can fail if the template contains unexpected placeholders
        
        # Use a safer, more direct approach
        prompt = "Extract metadata from the following text. Return a JSON object with fields: magazine, magazine_no, author, title, abstract, theme, format, geographic_area, keywords.\n\n"
        prompt += text_content
            
        # Check if response_format is supported for this model
        response_format_models = ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo", "gpt-4o", "gpt-4-turbo-preview", "gpt-4o-mini"]
        model_base = model.split('-')[0] + '-' + model.split('-')[1]  # Get base model name
        
        # Make the API call based on model support
        if model in response_format_models or model_base in response_format_models:
            print("Making API call with response_format=json")
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that extracts metadata from text. Return valid JSON with fields: magazine, magazine_no, author, title, abstract, theme, format, geographic_area, keywords. Ensure all fields are included."},
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
                    {"role": "system", "content": "You are a helpful assistant that extracts metadata from text. Return only valid JSON with fields: magazine, magazine_no, author, title, abstract, theme, format, geographic_area, keywords. Ensure all fields are included."},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                max_tokens=max_tokens
            )
            
        print("Successfully received response from OpenAI API")
        
        # Get the content from the response
        content = response.choices[0].message.content
        print("Got response content from API")
        
        # Clean and extract JSON from the response
        json_content = extract_json_from_response(content)
        print(f"Cleaned JSON content: {json_content[:100]}...")
        
        try:
            # Parse the JSON
            data = json.loads(json_content)
            
            print("Successfully parsed JSON response")
            
            # Ensure all required fields are present
            required_fields = ['magazine', 'magazine_no', 'author', 'title', 'abstract', 'theme', 'format', 'geographic_area', 'keywords']
            for field in required_fields:
                if field not in data:
                    print(f"Missing field '{field}' in response, creating default")
                    if field == 'abstract':
                        data[field] = "No abstract available"
                    elif field == 'keywords':
                        data[field] = "unknown, missing, unspecified"
                    else:
                        data[field] = "Unknown"
            
            # Add the text content and file path to the data
            data['text_content'] = text_content
            data['full_path'] = file_path
            
            return data
            
        except json.JSONDecodeError as json_error:
            print(f"JSON parsing error: {json_error}")
            print(f"Problematic JSON: {json_content}")
            
            # Create a fallback response with error information
            error_data = {
                'magazine': "Error",
                'magazine_no': "Error",
                'author': "Error",
                'title': os.path.basename(file_path),
                'abstract': f"Error parsing JSON during extraction: {str(json_error)}",
                'theme': "error",
                'format': "error",
                'geographic_area': "error",
                'keywords': "error, extraction, failed",
                'text_content': text_content,
                'full_path': file_path
            }
            return error_data
            
    except Exception as e:
        print(f"Error extracting fields: {e}")
        traceback.print_exc()
        
        # Return error response
        return {
            'magazine': "Error",
            'magazine_no': "Error",
            'author': "Error",
            'title': os.path.basename(file_path) if file_path else "Unknown",
            'abstract': f"Error during extraction: {str(e)}",
            'theme': "error",
            'format': "error",
            'geographic_area': "error",
            'keywords': "error, extraction, failed",
            'text_content': text_content,
            'full_path': file_path
        }

def batch_extract_fields(file_data_list: List[Dict[str, Any]], 
                         config: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """
    Process a batch of files for field extraction.
    
    Args:
        file_data_list: List of file data dictionaries with 'text_content' key
        config: Optional configuration dictionary
        
    Returns:
        Updated list of file data dictionaries with extracted fields
    """
    if not config:
        config = load_config()
    
    results = []
    
    for file_data in file_data_list:
        # Skip files that already have all extracted fields
        if all(k in file_data for k in ['abstract', 'theme', 'format', 'geographic_area', 'keywords']):
            results.append(file_data)
            continue
        
        # Extract fields from text content
        if 'text_content' in file_data and file_data['text_content']:
            # Get the file path as a string
            file_path = file_data.get('full_path', '')
            
            # Call with the right parameters - file_path as string
            extracted_fields = extract_fields_from_text(file_path, file_data['text_content'], config)
            
            # Update the file data with extracted fields
            file_data.update(extracted_fields)
        
        results.append(file_data)
    
    return results

def process_file(file_path, config, phase=1):
    """
    Process a file by extracting text and applying AI extraction.
    
    Args:
        file_path: Path to the file
        config: Configuration dictionary
        phase: Processing phase (1=extraction, 2=reconciliation)
        
    Returns:
        Dict containing file data
    """
    try:
        # Get the basedir from config
        base_dir = config.get('base_dir', '')
        print(f"Processing file: {file_path} (Phase {phase})")
        
        # Extract text from file
        text_content = extract_text_from_file(file_path)
        print(f"Extracted {len(text_content)} characters from file")
        
        # Create file data with basic file info
        file_data = {
            'full_path': file_path,
            'file_name': os.path.basename(file_path),
            'text_content': text_content,
            'processing_complete': False,
            'error': None
        }
        
        if phase >= 1:
            # Phase 1: Extract fields
            print("Phase 1: Extracting fields from text")
            
            # Extract fields from text content
            if 'text_content' in file_data and file_data['text_content']:
                extracted_fields = extract_fields_from_text(file_path, file_data['text_content'], config)
                
                # Update the file data with extracted fields
                file_data.update(extracted_fields)
                print("Phase 1 complete: Fields extracted successfully")
            else:
                file_data['error'] = "No text content available for extraction"
                print("Phase 1 error: No text content available")
                
        if phase >= 2:
            # Phase 2: Reconcile metadata
            print("Phase 2: Reconciling metadata")
            
            # Check if the minimum required fields are present
            try:
                reconciled_data = reconcile_metadata(file_data, config)
                
                # Update the file data with reconciled fields
                file_data.update(reconciled_data)
                print("Phase 2 complete: Metadata reconciled successfully")
            except Exception as reconcile_error:
                file_data['error'] = f"Reconciliation error: {str(reconcile_error)}"
                print(f"Phase 2 error: {str(reconcile_error)}")
                
        # Mark processing as complete
        file_data['processing_complete'] = True
        
        return file_data
        
    except Exception as e:
        print(f"Error processing file: {e}")
        traceback.print_exc()
        
        # Return error response
        return {
            'full_path': file_path,
            'file_name': os.path.basename(file_path),
            'text_content': "",
            'processing_complete': False,
            'error': f"Processing error: {str(e)}",
            'abstract': f"Error processing file: {str(e)}",
            'theme': 'error',
            'format': 'error',
            'geographic_area': 'error',
            'keywords': 'error, processing, failed'
        }

def extract_text_from_file(file_path: str) -> str:
    """
    Extract text content from a file based on its extension.
    Currently supports .docx files.
    
    Args:
        file_path: Path to the file
        
    Returns:
        Extracted text content
    """
    try:
        file_ext = Path(file_path).suffix.lower()
        
        if file_ext == '.docx':
            # Handle DOCX files
            doc = docx.Document(file_path)
            
            # Extract text content from paragraphs
            paragraphs = []
            for para in doc.paragraphs:
                if para.text.strip():
                    paragraphs.append(para.text)
            
            # Join paragraphs with newlines
            text_content = "\n".join(paragraphs)
            return text_content
            
        elif file_ext == '.txt':
            # Handle plain text files
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()
        else:
            print(f"Unsupported file extension: {file_ext}")
            return f"[Unsupported file format: {file_ext}]"
            
    except Exception as e:
        print(f"Error extracting text from file {file_path}: {e}")
        return f"[Error extracting text: {str(e)}]" 