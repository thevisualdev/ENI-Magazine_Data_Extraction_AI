import os
import re
import zipfile
import tempfile
import yaml
import pandas as pd
import logging
from typing import Dict, List, Any, Tuple, Optional

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def load_config() -> Dict[str, Any]:
    """Load configuration from config.yaml file."""
    with open('config.yaml', 'r') as f:
        return yaml.safe_load(f)

def extract_metadata_from_path(file_path: str, config: Dict[str, Any]) -> Dict[str, str]:
    """
    Extract metadata from file path based on folder structure.
    Expected structure: Magazine_No/Author/file.docx
    
    This function is designed to be flexible and handle exceptions in the folder structure.
    """
    # Get configurations
    magazine_pattern = config['folder_structure']['magazine_pattern']
    alternative_patterns = config['folder_structure'].get('alternative_patterns', [])
    known_magazines = config['folder_structure'].get('known_magazines', [])
    
    # Set up logging level from config
    log_level = config['folder_structure'].get('log_level', 'INFO')
    logging.getLogger().setLevel(getattr(logging, log_level))
    
    # Extract path components
    parts = file_path.split(os.sep)
    print(f"Processing file path: {file_path}")
    print(f"Path components: {parts}")
    
    # Extract base filename without extension and clean it up
    base_filename = os.path.splitext(os.path.basename(file_path))[0]
    # Remove problematic patterns from filenames
    for pattern in ["(Copia in conflitto di", "(copia in conflitto di", "Copia in conflitto di"]:
        if pattern in base_filename:
            # Get the author part before the problematic pattern
            base_filename = base_filename.split(pattern)[0].strip()
            break
    
    # Initialize metadata with default values
    metadata = {
        'magazine': 'Unknown',
        'magazine_no': '',
        'author': 'Unknown',
        'title': base_filename,
        'file_path': file_path
    }
    
    # IMPROVED: Direct extraction from known folder structure patterns
    if 'data' in parts:
        data_index = parts.index('data')
        
        # Check for magazine folder (like "WE" or "Orizzonti") after 'data'
        if len(parts) > data_index + 1:
            magazine_dir = parts[data_index + 1]
            
            # Match against known magazines from config
            magazine_name = None
            for mag_info in known_magazines:
                if magazine_dir.lower() == mag_info['name'].lower() or magazine_dir.lower() in [alias.lower() for alias in mag_info.get('aliases', [])]:
                    magazine_name = mag_info['name']
                    metadata['magazine'] = magazine_name
                    break
            
            if metadata['magazine'] == 'Unknown':
                metadata['magazine'] = magazine_dir  # Fallback to directory name
            
            # Extract issue number from next component
            if len(parts) > data_index + 2:
                issue_dir = parts[data_index + 2]
                
                # Try different patterns for issue number extraction
                
                # Pattern: "Orizzonti_62_x_web"
                x_web_match = re.search(r'.*_(\d+)_x_web$', issue_dir)
                if x_web_match:
                    metadata['magazine_no'] = x_web_match.group(1)
                
                # Pattern: "WE_60"
                if not metadata['magazine_no']:
                    underscore_match = re.search(r'.*_(\d+)$', issue_dir)
                    if underscore_match:
                        metadata['magazine_no'] = underscore_match.group(1)
                
                # Pattern: "We 54"
                if not metadata['magazine_no']:
                    space_match = re.search(r'.*\s+(\d+)$', issue_dir)
                    if space_match:
                        metadata['magazine_no'] = space_match.group(1)
                
                # Try to extract author from next component if available
                if len(parts) > data_index + 3:
                    author_dir = parts[data_index + 3]
                    # Skip if it looks like a number or special format
                    if not author_dir.isdigit() and "_" not in author_dir:
                        metadata['author'] = author_dir
    
    # If direct extraction didn't work, fallback to pattern matching
    if metadata['magazine'] == 'Unknown' or not metadata['magazine_no']:
        # Original pattern matching code (keep as fallback)
        magazine_found = False
        
        # First try with main pattern
        for part in parts:
            match = re.match(magazine_pattern, part)
            if match:
                metadata['magazine'] = match.group(1)
                metadata['magazine_no'] = match.group(2)
                magazine_found = True
                break
        
        # If no match found, try alternative patterns
        if not magazine_found and alternative_patterns:
            for part in parts:
                for pattern_dict in alternative_patterns:
                    pattern = pattern_dict.get('pattern', '')
                    if not pattern:
                        continue
                    
                    match = re.match(pattern, part)
                    if match:
                        # If pattern has two groups, use both magazine and number
                        if len(match.groups()) >= 2:
                            metadata['magazine'] = match.group(1)
                            metadata['magazine_no'] = match.group(2)
                        # If pattern has only one group, use it for magazine name
                        elif len(match.groups()) == 1:
                            metadata['magazine'] = match.group(1)
                        
                        magazine_found = True
                        break
                
                if magazine_found:
                    break
    
    # Try to extract additional information from filename if available
    filename = metadata['title']
    
    # Check if the filename contains author information (e.g., "Author_Title")
    if '_' in filename and metadata['author'] == 'Unknown':
        possible_author = filename.split('_')[0]
        # If this looks like an author name (not a date or number), use it
        if possible_author.isalpha():
            metadata['author'] = possible_author
            # Update title to remove the author part
            metadata['title'] = '_'.join(filename.split('_')[1:])
    
    # Normalize magazine names using known_magazines list
    if known_magazines:
        magazine_lower = metadata['magazine'].lower()
        for magazine_info in known_magazines:
            if magazine_lower == magazine_info['name'].lower() or magazine_lower in magazine_info.get('aliases', []):
                metadata['magazine'] = magazine_info['name']
                break
    
    print(f"Extracted metadata: Magazine='{metadata['magazine']}', Issue='{metadata['magazine_no']}', Author='{metadata['author']}'")
    
    # Log any unusual structure for review
    if config['folder_structure'].get('log_unknown', False):
        if metadata['magazine'] == 'Unknown' or metadata['author'] == 'Unknown':
            logger.warning(f"Could not extract complete metadata from path: {file_path}")
            logger.warning(f"Extracted metadata: {metadata}")
    
    return metadata

def process_zip_file(zip_file, config: Dict[str, Any]) -> List[Dict[str, str]]:
    """
    Process a ZIP file containing the folder structure.
    Returns a list of dictionaries with metadata for each DOCX file.
    """
    file_metadata_list = []
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Extract the zip file
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)
        
        # Walk through the directory structure
        for root, _, files in os.walk(temp_dir):
            for file in files:
                # Check for DOCX and other potential document formats
                if file.lower().endswith(('.docx', '.doc')):
                    file_path = os.path.join(root, file)
                    
                    # Instead of calculating a relative path, use the full file path
                    metadata = extract_metadata_from_path(file_path, config)
                    metadata['full_path'] = file_path  # Store the full path for later use
                    
                    # Add to the list
                    file_metadata_list.append(metadata)
    
    # Log summary
    logger.info(f"Processed ZIP file. Found {len(file_metadata_list)} document files.")
    
    return file_metadata_list

def process_directory(directory_path: str, config: Dict[str, Any]) -> List[Dict[str, str]]:
    """
    Process a directory containing the folder structure.
    Returns a list of dictionaries with metadata for each DOCX file.
    """
    file_metadata_list = []
    
    # Validate that the directory exists
    if not os.path.isdir(directory_path):
        raise ValueError(f"Directory does not exist: {directory_path}")
    
    # Walk through the directory structure
    for root, _, files in os.walk(directory_path):
        # Skip hidden directories like .AppleDouble that contain macOS system files
        if '.AppleDouble' in root or '/.' in root:
            continue
            
        for file in files:
            # Check for DOCX and other potential document formats
            if file.lower().endswith(('.docx', '.doc')):
                file_path = os.path.join(root, file)
                
                # Instead of using relative path, pass the full file path
                metadata = extract_metadata_from_path(file_path, config)
                metadata['full_path'] = file_path  # Store the full path for later use
                
                # Add to the list
                file_metadata_list.append(metadata)
    
    # Log summary
    logger.info(f"Processed directory {directory_path}. Found {len(file_metadata_list)} document files.")
    
    # Check if no files were found - this might indicate a problem
    if len(file_metadata_list) == 0:
        logger.warning(f"No document files found in directory: {directory_path}")
        
        # Try to suggest what might be wrong if no files were found
        if not any(f.lower().endswith(('.docx', '.doc')) for _, _, files in os.walk(directory_path) for f in files):
            logger.warning("No .docx or .doc files found in any subdirectory.")
        else:
            logger.warning("Document files exist but could not be processed. Check the folder structure.")
    
    return file_metadata_list

def get_magazine_info(directory_name: str, config: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
    """
    Extract magazine name and number from a directory name.
    Returns (magazine_name, magazine_number) or (None, None) if no match.
    
    This function tries both the main pattern and all alternative patterns.
    """
    # Get the main pattern from config
    magazine_pattern = config['folder_structure']['magazine_pattern']
    
    # Try the main pattern first
    match = re.match(magazine_pattern, directory_name)
    if match:
        return match.group(1), match.group(2)
    
    # If no match, try alternative patterns
    alternative_patterns = config['folder_structure'].get('alternative_patterns', [])
    for pattern_dict in alternative_patterns:
        pattern = pattern_dict.get('pattern', '')
        if not pattern:
            continue
        
        match = re.match(pattern, directory_name)
        if match:
            # If pattern has two groups, return both
            if len(match.groups()) >= 2:
                return match.group(1), match.group(2)
            # If pattern has only one group, return it as magazine name and empty string for number
            elif len(match.groups()) == 1:
                return match.group(1), ""
    
    return None, None 