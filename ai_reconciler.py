import json
import yaml
import os
from typing import Dict, Any, Optional
from openai import OpenAI
import dotenv
import traceback

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
                "magazine": "Unknown",
                "magazine_no": "Unknown",
                "author": "Unknown",
                "title": "Unknown",
                "language": "Unknown",
                "abstract_ita": "Impossibile analizzare la risposta JSON",
                "abstract_eng": "Could not parse JSON response",
                "theme": "Unknown",
                "format": "Unknown",
                "geographic_area": "Unknown",
                "keywords": "errore, analisi, fallita"
            }
            
            # Try to extract field values with regex
            fields = ["magazine", "magazine_no", "author", "title", "language", 
                     "abstract_ita", "abstract_eng", "theme", "format", 
                     "geographic_area", "keywords"]
            
            for field in fields:
                # Try to extract the field with a simple pattern
                # This handles both "field": "value" and 'field': 'value' formats
                pattern = r'["\']' + field + r'["\']\s*:\s*["\']([^"\']*)["\']'
                match = re.search(pattern, content)
                if match:
                    fallback[field] = match.group(1)
            
            print(f"Created fallback JSON with {sum(1 for v in fallback.values() if v)} fields")
            return json.dumps(fallback)

def extract_json_from_response(content):
    """
    Extract and clean JSON from the OpenAI API response for reconciliation.
    
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
        except json.JSONDecodeError:
            # If that fails, make some simple fixes
            try:
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
                except json.JSONDecodeError:
                    # One last attempt - rebuild the entire JSON structure
                    fallback = {
                        "magazine": "Unknown",
                        "magazine_no": "Unknown",
                        "author": "Unknown",
                        "title": "Unknown",
                        "language": "Unknown",
                        "abstract_ita": "Impossibile analizzare la risposta JSON",
                        "abstract_eng": "Could not parse JSON response",
                        "theme": "Unknown",
                        "format": "Unknown",
                        "geographic_area": "Unknown",
                        "keywords": "errore, analisi, fallita"
                    }
                    
                    # Try to extract field values with regex
                    fields = ["magazine", "magazine_no", "author", "title", "language", 
                            "abstract_ita", "abstract_eng", "theme", "format", 
                            "geographic_area", "keywords"]
                    
                    for field in fields:
                        # Match patterns like "field": "value" with various quote combinations
                        patterns = [
                            r'"' + field + r'":\s*"([^"]*)"',  # "field": "value"
                            r'"' + field + r'":\s*\'([^\']*)\'',  # "field": 'value'
                            r'\'' + field + r'\':\s*"([^"]*)"',  # 'field': "value"
                            r'\'' + field + r'\':\s*\'([^\']*)\'',  # 'field': 'value'
                            r'"' + field + r'":\s*([^,"\'}\n]*)[\n,}]',  # "field": value
                            r'\'' + field + r'\':\s*([^,"\'}\n]*)[\n,}]'  # 'field': value
                        ]
                        
                        for pattern in patterns:
                            match = re.search(pattern, content)
                            if match:
                                fallback[field] = match.group(1).strip()
                                break
                    
                    return json.dumps(fallback)
            except Exception as e:
                print(f"Error trying to clean JSON: {e}")
                # If all cleanup attempts fail, return empty JSON
                return "{}"
                
    except Exception as e:
        print(f"Error in extract_json_from_response: {e}")
        return "{}"

def reconcile_metadata(file_data: Dict[str, Any], config: Dict[str, Any] = None) -> Dict[str, str]:
    """
    Reconcile metadata extracted from folder structure and content using OpenAI API.
    Takes extracted metadata and article content, returns corrected/validated metadata.
    """
    if config is None:
        # Load default configuration
        config = load_config()

    try:
        # Extract necessary data from file_data
        file_path = file_data.get('full_path', '')
        magazine = file_data.get('magazine', '')
        magazine_no = file_data.get('magazine_no', '')
        author = file_data.get('author', '')
        title = file_data.get('title', '')
        text_content = file_data.get('text_content', '')

        if not text_content:
            print(f"No text content found for {file_path}, skipping reconciliation")
            return file_data

        # Truncate text_content if it's too long (to fit within token limits)
        text_content = text_content[:8000]

        # Get API configuration
        model = config.get('openai', {}).get('model', '')
        if not model:
            model = "gpt-3.5-turbo"  # Default model
            
        # Get the reconciliation prompt template
        prompt_template = config.get('prompts', {}).get('reconcile_metadata', '')
        
        # If no template provided, use a fallback template
        if not prompt_template:
            prompt_template = """
            You are an expert data validator for ENI Magazine, tasked with reconciling metadata.

            You have been provided with:
            1. The file path: "{file_path}"
            2. Preliminary metadata extracted from the folder structure:
            - Magazine: "{magazine}"
            - Magazine Number: "{magazine_no}"
            - Author: "{author}"
            - Title: "{title}"
            
            Article text: {text_content}
            
            Return a JSON with corrected fields: magazine, magazine_no, author, title, language, abstract_ita, abstract_eng, theme, format, geographic_area, keywords.
            """
            print("Using fallback reconciliation prompt template")
        
        # Before formatting, replace any unescaped curly braces in the example JSON
        # This is needed to prevent format() from treating them as replacement fields
        if "{" in prompt_template:
            # First, temporarily replace our actual format variables with placeholders
            safe_fields = {}
            for field in ["file_path", "magazine", "magazine_no", "author", "title", "text_content"]:
                field_pattern = "{" + field + "}"
                safe_placeholder = f"__SAFE_FIELD_{field}__"
                safe_fields[field] = safe_placeholder
                prompt_template = prompt_template.replace(field_pattern, safe_placeholder)
            
            # Now escape all remaining curly braces (which are part of JSON examples)
            prompt_template = prompt_template.replace("{", "{{").replace("}", "}}")
            
            # Finally, restore our format variables
            for field, placeholder in safe_fields.items():
                prompt_template = prompt_template.replace(placeholder, "{" + field + "}")
        
        # Format the prompt template
        try:
            prompt = prompt_template.format(
                file_path=file_path,
                magazine=magazine,
                magazine_no=magazine_no,
                author=author,
                title=title,
                text_content=text_content
            )
        except KeyError as e:
            print(f"Formatting error with key: {e}")
            print("This is likely due to unescaped curly braces in the prompt template.")
            
            # Use a fallback prompt template that doesn't have JSON examples
            fallback_template = """
            You are an expert data validator for ENI Magazine.
            
            Review this file: "{file_path}"
            
            Current metadata:
            - Magazine: {magazine}
            - Magazine Number: {magazine_no}
            - Author: {author}
            - Title: {title}
            
            Based on the following content, verify and correct the metadata:
            
            {text_content}
            
            Return ONLY a valid JSON object with these fields: magazine, magazine_no, author, title, 
            language, abstract_ita, abstract_eng, theme, format, geographic_area, keywords.
            """
            
            prompt = fallback_template.format(
                file_path=file_path,
                magazine=magazine,
                magazine_no=magazine_no,
                author=author,
                title=title,
                text_content=text_content[:4000]  # Limit content size in fallback
            )
            print("Using fallback prompt template without JSON examples")
        
        # Check if all metadata has been pre-filled or already exists
        if all(k in file_data and file_data[k] for k in ['language', 'abstract_ita', 'abstract_eng', 'theme', 'format', 'geographic_area', 'keywords']):
            # Skip calling AI if we already have all the data
            print(f"All metadata fields already exist for {file_path}, skipping reconciliation.")
            return file_data
        
        # Ensure we have a valid API key
        api_key = setup_openai_api(config)
        
        # Create OpenAI client
        client = create_openai_client(api_key)
        
        # Get OpenAI settings from config
        temperature = config['openai'].get('temperature', 0.1)
        max_tokens = config['openai'].get('max_tokens', 1000)
        
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
        
        # Extract data from response
        try:
            # Parse the JSON response
            data = json.loads(json_content)
            
            # Make a copy of the original file data to preserve any original fields
            reconciled_data = file_data.copy()
            
            # Add original metadata fields if not already present
            if 'original_magazine' not in reconciled_data:
                reconciled_data['original_magazine'] = file_data.get('magazine', '')
            if 'original_magazine_no' not in reconciled_data:
                reconciled_data['original_magazine_no'] = file_data.get('magazine_no', '')
            if 'original_author' not in reconciled_data:
                reconciled_data['original_author'] = file_data.get('author', '')
            if 'original_title' not in reconciled_data:
                reconciled_data['original_title'] = file_data.get('title', '')
            
            # Update with reconciled fields
            reconciled_data['magazine'] = data.get('magazine', file_data.get('magazine', ''))
            reconciled_data['magazine_no'] = data.get('magazine_no', file_data.get('magazine_no', ''))
            reconciled_data['author'] = data.get('author', file_data.get('author', ''))
            reconciled_data['title'] = data.get('title', file_data.get('title', ''))
            reconciled_data['language'] = data.get('language', file_data.get('language', 'Unknown'))
            reconciled_data['abstract_ita'] = data.get('abstract_ita', file_data.get('abstract_ita', ''))
            reconciled_data['abstract_eng'] = data.get('abstract_eng', file_data.get('abstract_eng', ''))
            reconciled_data['theme'] = data.get('theme', file_data.get('theme', ''))
            reconciled_data['format'] = data.get('format', file_data.get('format', ''))
            reconciled_data['geographic_area'] = data.get('geographic_area', file_data.get('geographic_area', ''))
            reconciled_data['keywords'] = data.get('keywords', file_data.get('keywords', ''))
            
            # For backward compatibility - update 'abstract' field with the appropriate language or combine both
            if reconciled_data.get('language', '') == 'ITA':
                reconciled_data['abstract'] = reconciled_data.get('abstract_ita', reconciled_data.get('abstract_eng', ''))
            else:
                reconciled_data['abstract'] = reconciled_data.get('abstract_eng', reconciled_data.get('abstract_ita', ''))
            
            # Preserve non-AI-parsed fields
            for key, value in file_data.items():
                if key not in reconciled_data and key != 'text_content':
                    reconciled_data[key] = value
            
            # Add the reconciled flag
            reconciled_data['reconciled'] = True
            
            return reconciled_data
                
        except json.JSONDecodeError as json_error:
            print(f"Error parsing JSON in reconciliation: {json_error}")
            print(f"Problematic JSON: {json_content}")
            
            # Return original data if JSON parsing fails
            file_data['reconciled'] = False
            file_data['reconciliation_error'] = f"JSON parsing error: {str(json_error)}"
            return file_data
            
    except Exception as e:
        print(f"Error in reconciliation process: {e}")
        traceback.print_exc()
        
        # Return original data with error flag
        file_data['reconciled'] = False
        file_data['reconciliation_error'] = f"Reconciliation error: {str(e)}"
        return file_data

def batch_reconcile_metadata(file_data_list: list, config: Optional[Dict[str, Any]] = None) -> list:
    """
    Batch reconcile metadata for multiple files.
    
    Args:
        file_data_list: List of file data dictionaries to reconcile
        config: Configuration dictionary
        
    Returns:
        List of file data dictionaries with reconciled metadata
    """
    results = []
    
    for file_data in file_data_list:
        # Skip if already reconciled
        if file_data.get('reconciled', False):
            results.append(file_data)
            continue
            
        # Skip if we don't have text content
        if 'text_content' not in file_data or not file_data['text_content']:
            file_data['reconciled'] = False
            file_data['reconciliation_error'] = "No text content available for reconciliation"
            results.append(file_data)
            continue
            
        # Check if we already have all the required fields filled in
        if all(k in file_data and file_data[k] for k in ['language', 'abstract_ita', 'abstract_eng', 'theme', 'format', 'geographic_area', 'keywords']):
            # Already has all metadata, just mark as reconciled
            file_data['reconciled'] = True
            results.append(file_data)
            continue
        
        # Reconcile metadata
        reconciled_data = reconcile_metadata(file_data, config)
        results.append(reconciled_data)
    
    return results 