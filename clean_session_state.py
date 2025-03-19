#!/usr/bin/env python3
"""
Session State Cleaner for Streamlit ENI Magazine App
This script removes base64 encoded images and other large data from the session state.
Run this when temp.state.json gets too large.
"""

import os
import json
import shutil
import glob
import time
from pprint import pprint
import tempfile

def main():
    print("ENI Magazine App - Session State Cleaner")
    print("----------------------------------------")
    
    # Find all session state files
    session_files = glob.glob("./.streamlit/session_state_*.json")
    temp_state_file = "./.streamlit/temp.state.json"
    
    if os.path.exists(temp_state_file):
        session_files.append(temp_state_file)
    
    if not session_files:
        print("No session state files found.")
        return
    
    print(f"Found {len(session_files)} session state files.")
    
    # Process each file
    for session_file in session_files:
        process_session_file(session_file)
    
    print("\nDone! Please restart the Streamlit app.")

def process_session_file(file_path):
    print(f"\nProcessing {file_path}...")
    
    # Check file size before
    size_before = os.path.getsize(file_path) / (1024 * 1024)  # Convert to MB
    print(f"File size before: {size_before:.2f} MB")
    
    try:
        # Read the session state
        with open(file_path, 'r') as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                print("Error decoding JSON. File may be corrupted.")
                return
        
        # Create a backup
        backup_path = f"{file_path}.bak.{int(time.time())}"
        shutil.copy2(file_path, backup_path)
        print(f"Created backup at {backup_path}")
        
        # Clean the data
        cleaned_data = clean_session_state(data)
        
        # Write back the cleaned session state
        with open(file_path, 'w') as f:
            json.dump(cleaned_data, f)
        
        # Check file size after
        size_after = os.path.getsize(file_path) / (1024 * 1024)  # Convert to MB
        print(f"File size after: {size_after:.2f} MB")
        print(f"Reduced by: {(size_before - size_after):.2f} MB ({((size_before - size_after) / size_before * 100):.1f}%)")
    
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")

def clean_session_state(data):
    """
    Remove base64 images and other large data from session state
    """
    if not isinstance(data, dict):
        return data
    
    # Handle the session state data structure
    if 'file_data_list' in data:
        cleaned_file_data = []
        for file_data in data['file_data_list']:
            cleaned_file = clean_file_data(file_data)
            cleaned_file_data.append(cleaned_file)
        data['file_data_list'] = cleaned_file_data
        print(f"Cleaned {len(cleaned_file_data)} file data entries.")
    
    # Clean any other session state data
    for key in list(data.keys()):
        if key == 'dataframe':
            # If there's a dataframe in the session state, remove it
            # It can be reconstructed from file_data_list
            del data[key]
            print("Removed dataframe from session state.")
        
        # Handle nested dictionaries recursively
        elif isinstance(data[key], dict):
            data[key] = clean_session_state(data[key])
    
    return data

def clean_file_data(file_data):
    """
    Clean a single file_data entry to remove base64 images and reduce memory usage
    """
    if not isinstance(file_data, dict):
        return file_data
    
    # Create a new clean file data
    clean_data = {}
    
    # Copy over essential data, excluding base64 and large text content
    for key, value in file_data.items():
        # Skip base64 preview images
        if key == 'preview_image' and isinstance(value, str) and value.startswith('data:image'):
            continue
            
        # Keep the path to the image instead
        elif key == 'preview_image_path':
            clean_data[key] = value
            
        # Remove text_content but record its length
        elif key == 'text_content' and isinstance(value, str):
            clean_data['text_content_length'] = len(value)
            # Keep just the first 100 chars to help with debugging
            clean_data['text_content_preview'] = value[:100] + '...' if len(value) > 100 else value
            
        # Keep everything else
        else:
            clean_data[key] = value
    
    return clean_data

if __name__ == "__main__":
    main() 