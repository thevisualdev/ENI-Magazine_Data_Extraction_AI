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
    
    # Initialize metadata with default values
    metadata = {
        'magazine': 'Unknown',
        'magazine_no': '',
        'author': 'Unknown',
        'title': os.path.splitext(os.path.basename(file_path))[0],
        'file_path': file_path
    }
    
    # Try to match magazine pattern in all path components, not just at a specific level
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
    
    # Try to extract author from context - check different possibilities
    if len(parts) >= 3:  # Standard structure: Magazine_No/Author/file.docx
        # Standard case - author is the second-to-last directory
        metadata['author'] = parts[-2]
    elif len(parts) >= 2:  # Simplified structure: Author/file.docx
        # Simplified case - author is the directory containing the file
        metadata['author'] = parts[-2]
    
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
                    # Calculate relative path from temp_dir
                    rel_path = os.path.relpath(file_path, temp_dir)
                    
                    # Extract metadata from path
                    metadata = extract_metadata_from_path(rel_path, config)
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
        for file in files:
            # Check for DOCX and other potential document formats
            if file.lower().endswith(('.docx', '.doc')):
                file_path = os.path.join(root, file)
                
                # Calculate relative path from directory_path
                rel_path = os.path.relpath(file_path, directory_path)
                
                # Extract metadata from path
                metadata = extract_metadata_from_path(rel_path, config)
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