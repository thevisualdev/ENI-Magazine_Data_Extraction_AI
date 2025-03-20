import os
import sys
import streamlit as st
import pandas as pd
import time
import yaml
import tempfile
import shutil
from PIL import Image
import io
import base64
import threading
import queue
from typing import List, Dict, Any, Optional, Tuple, Union, Set
import dotenv
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter, defaultdict
import re
from datetime import datetime
import asyncio
import json
import traceback
import openai  # Add the import for OpenAI
# Import visualization functions from our new module
from visualizations import (
    plot_magazine_distribution,
    plot_language_distribution,
    plot_theme_distribution,
    plot_theme_pie,
    plot_top_authors,
    plot_issue_distribution,
    plot_magazine_issues,
    plot_format_distribution,
    plot_format_pie,
    plot_theme_trends,
    show_overview_charts,
    plot_format_by_magazine,
    plot_theme_trends_plotly,
    build_keyword_network,
    render_keyword_network
)

# Load environment variables from .env file
dotenv.load_dotenv()

# Import custom modules
from file_manager import load_config, process_zip_file, process_directory
from docx_parser import extract_docx_content, save_preview_image, find_folder_images
from ai_extractor import extract_fields_from_text, batch_extract_fields, setup_openai_api
from ai_reconciler import reconcile_metadata, batch_reconcile_metadata
from csv_manager import create_dataframe, save_to_csv, read_csv

# Set page config
st.set_page_config(
    page_title="ENI Magazine Data Extraction",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Define the function before it's called
def check_background_updates():
    """Check if there are any updates from background threads"""
    try:
        if os.path.exists('temp_state.json'):
            with open('temp_state.json', 'r') as f:
                data = json.load(f)
            
            # Apply the updates to the session state
            updates = data.get('updates', {})
            for key, value in updates.items():
                if key in ['file_data_list']:
                    # Special handling for complex objects
                    st.session_state[key] = value
                else:
                    # Simple values
                    st.session_state[key] = value
            
            # Remove the temporary file
            os.remove('temp_state.json')
            
            # Force a rerun to display the updated state
            st.rerun()
    except Exception as e:
        print(f"Error checking for background updates: {str(e)}")

# Check for background updates at the start of each rerun
check_background_updates()

# Create a temporary directory for image storage
TEMP_DIR = os.path.join(tempfile.gettempdir(), "eni_magazine_images")
os.makedirs(TEMP_DIR, exist_ok=True)

# Define the output CSV path
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)
CSV_PATH = os.path.join(OUTPUT_DIR, "extracted_data.csv")

# Initialize session state variables
if 'file_data_list' not in st.session_state:
    st.session_state.file_data_list = []
if 'is_processing' not in st.session_state:
    st.session_state.is_processing = False
if 'processed_count' not in st.session_state:
    st.session_state.processed_count = 0
if 'current_processing_file' not in st.session_state:
    st.session_state.current_processing_file = None
if 'current_processing_phase' not in st.session_state:
    st.session_state.current_processing_phase = None
if 'stop_requested' not in st.session_state:
    st.session_state.stop_requested = False
if 'api_key_valid' not in st.session_state:
    st.session_state.api_key_valid = False
if 'processing_phase' not in st.session_state:
    st.session_state.processing_phase = 1  # Default to phase 1
if 'last_batch_success_count' not in st.session_state:
    st.session_state.last_batch_success_count = 0
if 'last_batch_error_count' not in st.session_state:
    st.session_state.last_batch_error_count = 0
if 'batch_completed' not in st.session_state:
    st.session_state.batch_completed = False
if 'total_queue_size' not in st.session_state:
    st.session_state.total_queue_size = 0
if 'progress_percentage' not in st.session_state:
    st.session_state.progress_percentage = 0.0
if 'processing_results' not in st.session_state:
    st.session_state.processing_results = {'success': 0, 'error': 0}
if 'app_mode' not in st.session_state:
    st.session_state.app_mode = "extraction"

# Global variables for batch processing
processing_queue = queue.Queue()
stop_processing = threading.Event()  # Used to signal stopping from main thread to background thread
processing_thread = None
batch_success_count = 0  # Thread-safe counter for successful files
batch_error_count = 0  # Thread-safe counter for failed files
current_file_name = None  # Thread-safe variable for current file name
current_phase = None  # Thread-safe variable for current phase
processed_count = 0  # Thread-safe counter for processed files
progress_percentage = 0.0  # Thread-safe variable for progress

def display_image(image_source):
    """Display an image from file path or base64 string in Streamlit."""
    if not image_source:
        return
        
    try:
        # Check if it's a file path
        if isinstance(image_source, str) and not image_source.startswith('data:'):
            if image_source == "[embedded image in document]":
                st.info("Image embedded in document")
                return
                
            if os.path.exists(image_source):
                # Load and display the image from file
                image = Image.open(image_source)
                image.thumbnail((200, 200), Image.LANCZOS)
                st.image(image, width=200)
                return
        
        # Fallback to base64 handling for legacy data
        if isinstance(image_source, str) and image_source.startswith('data:'):
            # Extract the image data
            image_data = image_source.split(',')[1]
            
            # Decode base64 data
            image_bytes = base64.b64decode(image_data)
            
            # Open image using PIL
            image = Image.open(io.BytesIO(image_bytes))
            
            # Resize image to reduce memory footprint
            image.thumbnail((200, 200), Image.LANCZOS)
            
            # Display the image with limited width
            st.image(image, width=200)
            return
            
        st.error("Unsupported image format")
    except Exception as e:
        st.error(f"Error displaying image: {e}")

def process_file(file_data: Dict[str, Any], df: pd.DataFrame, config: Dict[str, Any], reconcile: bool = False) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """
    Process a single file: extract content, AI fields, and update the DataFrame.
    With optional metadata reconciliation.
    
    Args:
        file_data: File data dictionary
        df: The DataFrame to update
        config: Configuration dictionary
        reconcile: Whether to perform metadata reconciliation (phase 2)
        
    Returns:
        Tuple of (updated file_data, updated DataFrame)
    """
    try:
        # Make a deep copy of the file_data to avoid modifying the original
        file_data_copy = file_data.copy()
        
        # Phase 1: Extract text and preview image from the DOCX file (if not already done)
        if 'text_content' not in file_data_copy or not file_data_copy['text_content']:
            text_content, preview_image_path = extract_docx_content(file_data_copy['full_path'])
            
            file_data_copy['text_content'] = text_content
            
            # Store the preview image path
            if preview_image_path:
                file_data_copy['preview_image_path'] = preview_image_path
            
            # Find images in the document folder and store their paths
            folder_images = find_folder_images(file_data_copy['full_path'])
            file_data_copy['folder_images'] = folder_images
            
            # Set preview_image_path if not already set
            if not preview_image_path and folder_images:
                file_data_copy['preview_image_path'] = folder_images[0]
            elif not preview_image_path:
                file_data_copy['preview_image_path'] = "[embedded image in document]"
        
        # Check if we need to run Phase 1 extraction
        phase1_needed = not all(k in file_data_copy for k in ['abstract', 'theme', 'format', 'geographic_area', 'keywords'])
        if phase1_needed or (not reconcile):
            # Extract fields using AI
            # If re-running Phase 1 after Phase 2, preserve original metadata fields
            original_fields = {}
            if all(k in file_data_copy for k in ['original_magazine', 'original_magazine_no', 'original_author', 'original_title']):
                # Save the original values
                original_fields = {
                    'original_magazine': file_data_copy.get('original_magazine'),
                    'original_magazine_no': file_data_copy.get('original_magazine_no'),
                    'original_author': file_data_copy.get('original_author'),
                    'original_title': file_data_copy.get('original_title')
                }
            
            # Get the file path
            file_path = file_data_copy.get('full_path', '')
            
            # Extract new fields
            extracted_fields = extract_fields_from_text(file_path, file_data_copy['text_content'], config)
            
            # Update file data with extracted fields
            file_data_copy.update(extracted_fields)
            
            # Ensure all required fields are present
            ensure_required_fields(file_data_copy)
        
        # Phase 2: Reconcile metadata (if requested)
        if reconcile:
            # Ensure we have the required fields before reconciliation
            ensure_required_fields(file_data_copy)
            
            # Run reconciliation
            reconciled_data = reconcile_metadata(file_data_copy, config)
            file_data_copy.update(reconciled_data)
        
        # Update the DataFrame
        df = update_dataframe(file_data_copy, df)
        
        return file_data_copy, df
    
    except Exception as e:
        print(f"Error processing file: {e}")
        # Return the original file_data and DataFrame if there's an error
        return file_data, df

def ensure_required_fields(file_data: Dict[str, Any]) -> None:
    """Ensure required field keys exist in the dictionary, but don't add default values."""
    required_fields = ['abstract', 'theme', 'format', 'geographic_area', 'keywords', 'language']
    for field in required_fields:
        if field not in file_data:
            # Only add the field as empty, don't make assumptions about values
            file_data[field] = ""
            print(f"Added missing field '{field}' as empty value")

def update_dataframe(file_data: Dict[str, Any], df: pd.DataFrame) -> pd.DataFrame:
    """
    Update DataFrame with file data.
    
    Args:
        file_data: File data dictionary
        df: DataFrame to update
        
    Returns:
        Updated DataFrame
    """
    # Check if the file is already in the DataFrame
    if 'full_path' not in file_data:
        return df
        
    file_path = file_data['full_path']
    file_row = df[df['full_path'] == file_path]
    
    if file_row.empty:
        # File not in DataFrame, add a new row
        df_row = pd.DataFrame([{
            'filename': file_data.get('filename', os.path.basename(file_path)),
            'full_path': file_path,
            'title': file_data.get('title', ''),
            'author': file_data.get('author', ''),
            'abstract': file_data.get('abstract', ''),
            'theme': file_data.get('theme', ''),
            'format': file_data.get('format', ''),
            'geographic_area': file_data.get('geographic_area', ''),
            'keywords': file_data.get('keywords', ''),
            'magazine': file_data.get('magazine', ''),
            'magazine_no': file_data.get('magazine_no', ''),
            'language': file_data.get('language', ''),
            'Last Updated': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'Phase': file_data.get('phase', 1)
        }])
        df = pd.concat([df, df_row], ignore_index=True)
    else:
        # Update the existing row
        for field in ['title', 'author', 'abstract', 'theme', 'format', 'geographic_area', 
                      'keywords', 'magazine', 'magazine_no', 'language']:
            if field in file_data:
                df.loc[df['full_path'] == file_path, field] = file_data[field]
                
        # Update timestamp
        df.loc[df['full_path'] == file_path, 'Last Updated'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    return df

def batch_processor(params=None):
    """Background worker thread to process files from the queue"""
    try:
        # Initialize key variables locally
        print(f"Starting batch processor")
        progress_percentage = 0.0
        
        # Extract parameters safely
        if params is None:
            params = {}
        
        api_key = params.get('api_key', None)
        file_data_list = params.get('file_data_list', [])
        total_queue_size = params.get('total_queue_size', 0)
            
        print(f"Batch processor started with {total_queue_size} files in queue")
        
        # Make sure we have an API key
        if not api_key:
            if 'OPENAI_API_KEY' in os.environ and os.environ['OPENAI_API_KEY']:
                api_key = os.environ['OPENAI_API_KEY']
                print(f"API key from environment: Found")
            else:
                print(f"API key from environment: Not found")
        
        if not api_key:
            print(f"WARNING: No API key found for batch processing!")
            print(f"CRITICAL ERROR: No API key available. Batch processing cannot continue.")
            return
        
        # Load the configuration and set API key
        config = load_config()
        config['openai']['api_key'] = api_key
        
        # Load or create the DataFrame for the CSV file
        try:
            csv_path = os.path.join("output", "extracted_data.csv")  # Use the correct path constant
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                if df.empty:
                    df = create_dataframe(file_data_list)
            else:
                df = create_dataframe(file_data_list)
                os.makedirs(os.path.dirname(csv_path), exist_ok=True)
                save_to_csv(df, csv_path)
        except Exception as e:
            print(f"Error loading/creating CSV: {str(e)}")
            df = create_dataframe(file_data_list)
        
        # Process files from the queue
        success_count = 0
        error_count = 0
        
        # Create a dictionary to map file IDs to their index in file_data_list
        file_id_map = {}
        for i, fd in enumerate(file_data_list):
            if 'id' in fd:
                file_id_map[fd['id']] = i
            elif 'full_path' in fd:
                file_id_map[fd['full_path']] = i
        
        while not stop_processing.is_set() and not processing_queue.empty():
            try:
                # Get the next file from the queue
                file_data, phase = processing_queue.get()
                
                filename = file_data.get('filename', file_data.get('title', 'Unknown file'))
                print(f"Processing {filename} (Phase {phase})")
                
                # Process the file
                start_time = time.time()
                
                # Use the existing process_file function that handles text extraction correctly
                reconcile = (phase == 2)
                updated_file_data, df = process_file(file_data, df, config, reconcile=reconcile)
                
                success = True  # If we got here without exception, consider it a success
                
                # Calculate duration
                duration = time.time() - start_time
                
                # Update file status in file_data_list if we can find it
                file_id = file_data.get('id')
                file_path = file_data.get('full_path')
                
                # Try to find the file in our mapping
                idx = None
                if file_id and file_id in file_id_map:
                    idx = file_id_map[file_id]
                elif file_path and file_path in file_id_map:
                    idx = file_id_map[file_path]
                
                if idx is not None:
                    file_data_list[idx] = updated_file_data
                
                # Update the DataFrame
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                
                # Check if the file is already in the DataFrame
                file_row = df[df['full_path'] == file_data.get('full_path', '')]
                if not file_row.empty:
                    df.loc[df['full_path'] == file_data.get('full_path', ''), 'Last Updated'] = timestamp
                    df.loc[df['full_path'] == file_data.get('full_path', ''), 'Duration'] = f"{duration:.2f}s"
                    df.loc[df['full_path'] == file_data.get('full_path', ''), 'Phase'] = phase
                
                # Update tracking variables
                if success:
                    success_count += 1
                else:
                    error_count += 1
            
                # Mark the task as done
                processing_queue.task_done()
            
                # Update progress
                processed_count = success_count + error_count
                if total_queue_size > 0:
                    progress_percentage = (processed_count / total_queue_size) * 100
                
                # Save the updated information periodically (every 10 files) or when queue is empty
                if processed_count % 10 == 0 or processing_queue.empty():
                    try:
                        # Save DataFrame to CSV
                        save_to_csv(df, csv_path)
                        
                        # Update session state variables
                        next_loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(next_loop)
                        next_loop.run_until_complete(update_session_state(processed_count, progress_percentage, file_data_list))
                        next_loop.close()
                    except Exception as e:
                        print(f"Error saving progress: {str(e)}")
            
            except Exception as e:
                print(f"Error processing file in batch: {str(e)}")
                error_count += 1
                processing_queue.task_done()
        
        # Final update to indicate completion
        try:
            # Save DataFrame to CSV one final time
            save_to_csv(df, csv_path)
            
            # Update session state one final time
            next_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(next_loop)
            next_loop.run_until_complete(update_session_state(processed_count, progress_percentage, file_data_list))
            next_loop.close()
            
            print(f"Batch processing completed: {success_count} succeeded, {error_count} failed")
            
        except Exception as e:
            print(f"Error finalizing batch process: {str(e)}")
            
    except Exception as e:
        print(f"Critical error in batch processor: {str(e)}")
        traceback.print_exc()

async def update_session_state(processed_count, progress_percentage, file_data_list, is_done=False):
    """Update session state from the background thread using asyncio"""
    try:
        # Use a lock or queue to update the session state
        # This is a simple implementation using set_page_config which allows background threads to communicate with the main app
        # Create a dummy query parameter to force a rerun
        rerun_id = int(time.time() * 1000)
        query_params = {"_update": str(rerun_id)}
        
        # Create lightweight version of file_data_list to reduce session state size
        lightweight_data = create_lightweight_file_data(file_data_list)
        
        # Serialize the data we want to update
        state_updates = {
            "processed_count": processed_count,
            "progress_percentage": progress_percentage,
            "file_data_list": lightweight_data
        }
        
        if is_done:
            state_updates["is_processing"] = False
        
        # Store in a temporary file for the main thread to pick up
        with open('temp_state.json', 'w') as f:
            json.dump({
                "timestamp": rerun_id,
                "updates": state_updates
            }, f, default=lambda o: str(o))
        
        print(f"Updated session state: {processed_count} files processed, {progress_percentage:.1f}% complete")
    except Exception as e:
        print(f"Error updating session state: {str(e)}")

def process_phase1(file_data, api_key):
    """Process a file for phase 1 - extract basic metadata"""
    try:
        # Implement your Phase 1 processing logic here
        # For example, extract metadata from the file using OpenAI
        
        # Set OpenAI API key for this request
        openai.api_key = api_key
        
        # Get contents for processing
        text_content = file_data.get('text_content', '')
        if not text_content:
            print(f"No text content found for {file_data.get('filename', file_data.get('title', 'Unknown'))}")
            return False
        
        # Example: Extract metadata using OpenAI
        # This is a placeholder - implement your actual extraction logic
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an assistant that extracts document metadata."},
                {"role": "user", "content": f"Extract the following from this document: abstract, theme, format, geographic area, and keywords. Format as JSON.\n\n{text_content[:4000]}"}
            ],
            temperature=0.3,
            max_tokens=500
        )
        
        # Process the response
        result = response.choices[0].message.content
        
        # Try to parse as JSON
        try:
            metadata = json.loads(result)
            
            # Update the file_data with extracted information
            if isinstance(metadata, dict):
                for key in ['abstract', 'theme', 'format', 'geographic_area', 'keywords']:
                    if key in metadata:
                        file_data[key] = metadata[key]
            
            return True
        except json.JSONDecodeError:
            print(f"Could not parse JSON from response for {file_data.get('filename', file_data.get('title', 'Unknown'))}")
            return False
    
    except Exception as e:
        print(f"Error in process_phase1: {str(e)}")
        return False

def process_phase2(file_data, api_key):
    """Process a file for phase 2 - deeper content analysis"""
    try:
        # Implement your Phase 2 processing logic here
        # For example, classify or categorize the file
        
        # Set OpenAI API key for this request
        openai.api_key = api_key
        
        # Get contents for processing
        text_content = file_data.get('text_content', '')
        if not text_content:
            print(f"No text content found for {file_data.get('filename', file_data.get('title', 'Unknown'))}")
            return False
        
        # Example: Perform deeper analysis using OpenAI
        # This is a placeholder - implement your actual analysis logic
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an assistant that performs deep document analysis."},
                {"role": "user", "content": f"Analyze this document and provide: main entities, key topics, sentiment, and importance rating (1-10). Format as JSON.\n\n{text_content[:4000]}"}
            ],
            temperature=0.3,
            max_tokens=500
        )
        
        # Process the response
        result = response.choices[0].message.content
        
        # Try to parse as JSON
        try:
            analysis = json.loads(result)
            
            # Update the file_data with analysis information
            if isinstance(analysis, dict):
                for key in ['entities', 'topics', 'sentiment', 'importance']:
                    if key in analysis:
                        file_data[key] = analysis[key]
            
            return True
        except json.JSONDecodeError:
            print(f"Could not parse JSON from response for {file_data.get('filename', file_data.get('title', 'Unknown'))}")
            return False
    
    except Exception as e:
        print(f"Error in process_phase2: {str(e)}")
        return False

# Function to update UI from thread-safe variables
def update_ui_from_processing_thread():
    """Update UI elements from thread-safe batch processing variables"""
    if processing_thread and processing_thread.is_alive():
        # Copy thread-safe variables to session state
        st.session_state.processed_count = processed_count
        st.session_state.progress_percentage = progress_percentage
        st.session_state.current_processing_file = current_file_name
        st.session_state.current_processing_phase = current_phase
        st.session_state.last_batch_success_count = batch_success_count
        st.session_state.last_batch_error_count = batch_error_count
    
    # Check if thread has finished
    if st.session_state.is_processing and (not processing_thread or not processing_thread.is_alive()):
        st.session_state.is_processing = False
        st.session_state.current_processing_file = None
        st.session_state.current_processing_phase = None
        st.session_state.batch_completed = True
        
        # Force reload of file_data_list from CSV to ensure UI is consistent
        if os.path.exists(CSV_PATH):
            df = read_csv(CSV_PATH)
            if df is not None:
                # Update file_data_list from CSV rows
                for i, file_data in enumerate(st.session_state.file_data_list):
                    # Find the matching row in the dataframe by full_path
                    matching_rows = df[df['full_path'] == file_data['full_path']]
                    if not matching_rows.empty:
                        row = matching_rows.iloc[0]
                        # Update the file data with the values from the CSV
                        for col in df.columns:
                            if col in ['abstract', 'abstract_ita', 'abstract_eng', 'language', 'theme', 'format', 
                                     'geographic_area', 'keywords', 'original_magazine', 'original_magazine_no', 
                                     'original_author', 'original_title', 'magazine', 'magazine_no', 'author', 
                                     'title', 'preview_image_path', 'folder_images']:
                                file_data[col] = row[col]
                        st.session_state.file_data_list[i] = file_data
                        
                # Set dataframe in session state        
                st.session_state.dataframe = df
    else:
        # If we're loading the CSV for data analysis but processing is still ongoing,
        # preserve the processing state variables to prevent counter reset
        if processing_thread and processing_thread.is_alive():
            # Make sure to preserve these when reloading or showing CSV data
            preserved_states = {
                'processed_count': st.session_state.processed_count,
                'progress_percentage': st.session_state.progress_percentage,
                'last_batch_success_count': st.session_state.last_batch_success_count,
                'last_batch_error_count': st.session_state.last_batch_error_count,
                'is_processing': True
            }
            
            # Update the dataframe without resetting the processing state
            if os.path.exists(CSV_PATH):
                df = read_csv(CSV_PATH)
                if df is not None:
                    st.session_state.dataframe = df
                    
            # Restore the preserved state variables
            for key, value in preserved_states.items():
                st.session_state[key] = value

def validate_openai_api_key(api_key: str) -> bool:
    """
    Validate OpenAI API key by attempting a simple API call.
    
    Args:
        api_key: The OpenAI API key to validate
        
    Returns:
        bool: True if valid, False otherwise
    """
    try:
        if not api_key:
            return False
        
        # Import directly to ensure we're using the latest version
        from openai import OpenAI
        
        # Simple direct initialization
        client = OpenAI(api_key=api_key)
        
        # Try a minimal API call without response_format
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",  # Use a reliable model for validation
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Say 'API key is valid' in one short sentence."}
            ],
            max_tokens=20,
            temperature=0
        )
        
        # If we get here, the key is valid
        return True
    except Exception as e:
        print(f"API Key validation error: {e}")
        return False

# App title and description
st.title("ENI Magazine Data Extraction & Analysis")

# Create tabs for main navigation
tab_extraction, tab_analysis = st.tabs(["Data Extraction", "Data Analysis"])

with tab_extraction:
    st.markdown("""
    This application extracts structured data from ENI Magazine DOCX files using a dual-phase approach:

    **Phase 1**: Initial extraction of metadata from folder structure and content using AI.  
    **Phase 2**: AI verification and reconciliation of metadata against article content.

    Upload a ZIP file containing the folder structure, or select a directory to process.
    """)
    
    # Load configuration
    config = load_config()

    # Sidebar for API key and file upload
    with st.sidebar:
        st.header("Configuration")
        
        # OpenAI API Key input (prominently at the top)
        st.subheader("OpenAI API Key")
        
        # Get API key from environment first, then config, then empty
        default_api_key = os.environ.get('OPENAI_API_KEY', config['openai'].get('api_key', ''))
        
        api_key = st.text_input(
            "Enter your OpenAI API key",
            value=default_api_key,
            type="password",
            help="Your OpenAI API key is required for AI extraction. For security, we recommend using environment variables instead of storing in config.yaml."
        )
        
        # Check if API key changed and save it to session state (not to config file)
        if api_key != config['openai'].get('api_key', ''):
            # Don't save API key to config file anymore
            st.session_state.api_key = api_key
            
            # Validate the API key
            is_valid = validate_openai_api_key(api_key)
            st.session_state.api_key_valid = is_valid
            
            if api_key and is_valid:
                st.success("✅ API Key saved and validated")
            elif api_key:
                st.error("❌ Invalid API Key")
            else:
                st.warning("⚠️ API Key required for AI extraction")
        
        # Display validation status if already checked
        elif 'api_key_valid' in st.session_state:
            if api_key and st.session_state.api_key_valid:
                st.success("✅ API Key valid")
            elif api_key:
                st.error("❌ Invalid API Key")
            else:
                st.warning("⚠️ API Key required for AI extraction")
        
        # Show advanced settings
        with st.expander("Advanced Settings"):
            st.subheader("OpenAI Model Settings")
            st.json({
                "model": config['openai']['model'],
                "temperature": config['openai']['temperature'],
                "max_tokens": config['openai']['max_tokens']
            })
            
            # Display the extraction prompt
            st.subheader("Phase 1: Extraction Prompt")
            st.text_area("Prompt Template", 
                       value=config['prompts']['extract_fields'],
                       height=200,
                       disabled=True)
            
            # Display the reconciliation prompt
            st.subheader("Phase 2: Reconciliation Prompt")
            st.text_area("Reconciliation Template", 
                       value=config['prompts']['reconcile_metadata'],
                       height=200,
                       disabled=True)

        # Add CSV management section to sidebar (if file_data_list exists)
        if 'file_data_list' in st.session_state and st.session_state.file_data_list:
            st.markdown("---")
            st.subheader("CSV Data Management")
            
            # Reset CSV button with warning
            if st.button("🗑️ Reset CSV Data", 
                     help="WARNING: This will reset all extracted data to initial state",
                     type="primary",
                     use_container_width=True):
                try:
                    # Show warning confirmation
                    if st.warning("⚠️ This will delete all extracted data! Are you sure?"):
                        # Remove the CSV file
                        if os.path.exists(CSV_PATH):
                            # Create backup before deletion
                            backup_path = f"{CSV_PATH}_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                            shutil.copy(CSV_PATH, backup_path)
                            os.remove(CSV_PATH)
                            
                        # Clear the dataframe from session state    
                        if 'dataframe' in st.session_state:
                            del st.session_state['dataframe']
                            
                        # Create a new DataFrame and save it
                        df = create_dataframe(st.session_state.file_data_list)
                        save_to_csv(df, CSV_PATH)
                        st.session_state.dataframe = df
                        
                        st.success("CSV data has been reset successfully. A backup was created.")
                        st.rerun()
                except Exception as e:
                    st.error(f"Error resetting CSV data: {e}")
        
        # File/Folder Upload Section
        st.header("Upload Files")
        
        # Add tabs for different upload methods - make "Select Directory" the default
        upload_tab2, upload_tab1 = st.tabs(["Select Directory", "Upload ZIP"])
        
        with upload_tab1:
            st.markdown("Upload a ZIP file containing the ENI magazine folder structure.")
            uploaded_file = st.file_uploader("Choose a ZIP file", type="zip")
            
            if uploaded_file is not None:
                with st.spinner("Processing ZIP file..."):
                    # Create a temporary file to store the uploaded ZIP
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.zip') as temp_file:
                        temp_file.write(uploaded_file.getvalue())
                        temp_zip_path = temp_file.name
                    
                    try:
                        # Process the ZIP file
                        config = load_config()
                        file_data_list = process_zip_file(temp_zip_path, config)
                        st.session_state.file_data_list = file_data_list
                        st.success(f"Processed ZIP file. Found {len(file_data_list)} DOCX files.")
                    except Exception as e:
                        st.error(f"Error processing ZIP file: {e}")
                    finally:
                        # Clean up the temporary file
                        if os.path.exists(temp_zip_path):
                            os.unlink(temp_zip_path)
        
        with upload_tab2:
            st.markdown("""
            Select a local directory containing the ENI magazine folder structure.
            
            **Expected structure**: 
            - Root folder contains magazine folders (e.g., `Orizzonti_55`)
            - Each magazine folder contains author folders
            - Each author folder contains DOCX files
            """)
            
            # Simpler layout with more prominence to the important elements
            directory_path = st.text_input(
                "Directory path",
                help="Enter the full path to the folder containing the magazine structure",
                key="directory_path_input"
            )
            
            # Better button layout with clear hierarchy
            col1, col2, col3 = st.columns([2, 1, 1])
            
            with col1:
                if directory_path:
                    if st.button("Process Directory", type="primary", use_container_width=True):
                        with st.spinner("Processing directory..."):
                            try:
                                config = load_config()
                                file_data_list = process_directory(directory_path, config)
                                st.session_state.file_data_list = file_data_list
                                st.success(f"Processed directory. Found {len(file_data_list)} DOCX files.")
                            except Exception as e:
                                st.error(f"Error processing directory: {e}")
                else:
                    st.button("Process Directory", disabled=True, use_container_width=True, 
                             help="Enter a directory path first")
            
            with col2:
                if st.button("Browse...", use_container_width=True):
                    try:
                        # Create a temporary command to run based on OS
                        if os.name == 'nt':  # Windows
                            command = "explorer"
                        elif os.name == 'posix':  # macOS or Linux
                            if 'darwin' in sys.platform:  # macOS
                                command = "open"
                            else:  # Linux
                                command = "xdg-open"
                        
                        # Run the file browser
                        os.system(f'{command} .')
                        
                        st.info("""
                        1. Navigate to your data directory
                        2. Copy the full path
                        3. Paste it in the text field above
                        
                        **Tip:** On macOS, right-click a folder and hold Option to see 'Copy as Pathname'
                        """)
                    except Exception as e:
                        st.error(f"Couldn't open file browser: {e}")
            
            with col3:
                if st.button("Examples", use_container_width=True):
                    st.info(f"""
                    Example paths:
                    - macOS: /Users/username/Documents/ENI_Magazines
                    - Windows: C:\\Users\\username\\Documents\\ENI_Magazines
                    - Linux: /home/username/ENI_Magazines
                    
                    **Current directory:** {os.getcwd()}
                    """)
            
            st.markdown("The application will try to handle exceptions in the folder structure.")
        
        # Batch processing controls
        if st.session_state.file_data_list:
            st.markdown("---")
            st.header("Batch Processing")
            
            # Check if API key is valid before allowing processing
            api_key_valid = st.session_state.get('api_key_valid', False) or validate_openai_api_key(config['openai']['api_key'])
            st.session_state.api_key_valid = api_key_valid
            
            # Select processing phase
            st.radio(
                "Select processing phase:",
                ["Phase 1: Initial Extraction", "Phase 2: Metadata Reconciliation", "Both Phases"],
                key="processing_phase_selection",
                index=0,
                help="Phase 1 extracts metadata from folder structure and content. Phase 2 reconciles metadata with AI."
            )
            
            # Map radio button selection to processing phase number
            phase_mapping = {
                "Phase 1: Initial Extraction": 1,
                "Phase 2: Metadata Reconciliation": 2,
                "Both Phases": 0  # 0 means both phases
            }
            st.session_state.processing_phase = phase_mapping[st.session_state.processing_phase_selection]
            
            # Define the batch processing function here before it's called
            def start_batch_processing():
                """Function to handle starting batch processing"""
                global processing_thread, stop_processing
                
                # Reset the stop flag
                stop_processing.clear()
                
                # Clear any existing queue
                while not processing_queue.empty():
                    try:
                        processing_queue.get_nowait()
                        processing_queue.task_done()
                    except:
                        pass
                
                # Ensure the OpenAI API key is set in the environment
                # This ensures the background thread will have access to it
                if 'api_key' in st.session_state and st.session_state.api_key:
                    os.environ['OPENAI_API_KEY'] = st.session_state.api_key
                    print(f"API key set in environment before starting batch processor")
                
                # Start batch processing
                st.session_state.is_processing = True
                st.session_state.processed_count = 0
                st.session_state.progress_percentage = 0.0
                
                selected_phase = st.session_state.processing_phase
                
                # Create deep copies of the file data list to prevent session state threading issues
                file_data_list = []
                if hasattr(st.session_state, 'file_data_list') and st.session_state.file_data_list:
                    # This creates a deep copy of each file data dictionary
                    file_data_list = [dict(item) for item in st.session_state.file_data_list]
                
                # Ensure the file_data_list is populated
                if not file_data_list:
                    st.error("No files found for processing. Please upload some files first.")
                    st.session_state.is_processing = False
                    return
                
                # Count files that need processing
                phase1_count = 0
                phase2_count = 0
                
                # Put all unprocessed files in the queue based on selected phase
                for file_data in file_data_list:
                    if selected_phase == 0 or selected_phase == 1:  # Phase 1 or Both
                        if not all(k in file_data for k in ['abstract', 'theme', 'format', 'geographic_area', 'keywords']):
                            # Create a deep copy to avoid thread issues
                            processing_queue.put((dict(file_data), 1))  # Phase 1
                            phase1_count += 1
                        
                    if selected_phase == 0 or selected_phase == 2:  # Phase 2 or Both
                        # For Phase 2, we need files that have completed Phase 1
                        if all(k in file_data for k in ['abstract', 'theme', 'format', 'geographic_area', 'keywords']):
                            # Create a deep copy to avoid thread issues
                            processing_queue.put((dict(file_data), 2))  # Phase 2
                            phase2_count += 1
                
                total_queue_size = phase1_count + phase2_count
                st.session_state.total_queue_size = total_queue_size
                
                if total_queue_size == 0:
                    st.warning("No files need processing for the selected phase. Try selecting a different phase.")
                    st.session_state.is_processing = False
                    return
                
                # Show what's being processed
                if selected_phase == 0:
                    st.info(f"Processing {phase1_count} files in Phase 1 and {phase2_count} files in Phase 2.")
                elif selected_phase == 1:
                    st.info(f"Processing {phase1_count} files in Phase 1.")
                elif selected_phase == 2:
                    st.info(f"Processing {phase2_count} files in Phase 2.")
                
                # Define batch processing parameters for thread
                batch_params = {
                    'api_key': os.environ.get('OPENAI_API_KEY', ''),
                    'total_queue_size': total_queue_size,
                    'file_data_list': file_data_list
                }
                
                # Start the background thread with the parameters
                processing_thread = threading.Thread(
                    target=batch_processor,
                    args=(batch_params,)
                )
                processing_thread.daemon = True
                processing_thread.start()
            
            if not api_key_valid:
                st.error("Please enter a valid OpenAI API key before processing files")
            elif st.session_state.is_processing:
                if st.button("Stop Processing"):
                    stop_processing.set()  # Signal the thread to stop
                    st.warning("Stopping after current file completes...")
            else:
                if st.button("Start Batch Processing"):
                    start_batch_processing()
            
            # Display progress information
            total_files = len(st.session_state.file_data_list)
            processed_files = st.session_state.processed_count
            total_queue_size = getattr(st.session_state, 'total_queue_size', 0)
            
            if st.session_state.is_processing:
                # Show a prettier progress bar with more accurate information
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    # Use the more accurate progress percentage if available
                    progress_pct = getattr(st.session_state, 'progress_percentage', 0.0)
                    if progress_pct == 0.0 and total_queue_size > 0:
                        progress_pct = min(1.0, processed_files / total_queue_size)
                    
                    # Convert from percentage (0-100) to proportion (0-1)
                    progress_bar = st.progress(progress_pct / 100.0)
                
                with col2:
                    if total_queue_size > 0:
                        st.text(f"{processed_files} of {total_queue_size} files")
                        st.text(f"{progress_pct*100:.1f}% complete")
                    else:
                        st.text("Processing...")
                
                # Show processing status with more details
                success_count = getattr(st.session_state, 'last_batch_success_count', 0)
                error_count = getattr(st.session_state, 'last_batch_error_count', 0)
                status_text = f"Processed: {success_count} success, {error_count} errors"
                
                status_col1, status_col2 = st.columns([1, 1])
                with status_col1:
                    if st.session_state.current_processing_file:
                        st.text(f"File: {st.session_state.current_processing_file}")
                with status_col2:
                    if st.session_state.current_processing_phase:
                        st.text(f"{st.session_state.current_processing_phase} - {status_text}")
            else:
                # When not processing, show a summary of files that need processing
                phase1_files = [f for f in st.session_state.file_data_list 
                                if not all(k in f for k in ['abstract', 'theme', 'format', 'geographic_area', 'keywords'])]
                
                phase2_files = [f for f in st.session_state.file_data_list 
                               if all(k in f for k in ['abstract', 'theme', 'format', 'geographic_area', 'keywords']) and
                               ('original_magazine' not in f or 'original_title' not in f)]
                
                complete_files = [f for f in st.session_state.file_data_list 
                                 if all(k in f for k in ['abstract', 'theme', 'format', 'geographic_area', 'keywords']) and
                                 'original_magazine' in f and 'original_title' in f]
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Files Needing Phase 1", len(phase1_files))
                with col2:
                    st.metric("Files Needing Phase 2", len(phase2_files))
                with col3:
                    st.metric("Completed Files", len(complete_files))
                    
                # Show a completion message if batch processing just finished
                if st.session_state.batch_completed:
                    success_count = st.session_state.last_batch_success_count
                    error_count = st.session_state.last_batch_error_count
                    
                    if success_count > 0 or error_count > 0:
                        st.success(f"✅ Batch processing complete: {success_count} files processed successfully")
                        if error_count > 0:
                            st.warning(f"⚠️ {error_count} files had errors - check the console log for details")
                        
                        if st.button("Clear This Message"):
                            st.session_state.batch_completed = False
                            st.rerun()
                        
                        st.markdown("---")

# Main area - display files and data
if st.session_state.file_data_list:
    # Update UI from processing thread if running
    update_ui_from_processing_thread()
    
    st.header("Files")
    
    # Reload CSV button (only keep this one in the main area)
    col1, col2 = st.columns([1, 3])
    with col1:
        if st.button("🔄 Reload CSV Data", 
                  help="Load the latest data from the CSV file on disk",
                  type="secondary",
                  use_container_width=True):
            try:
                # Force reload from disk
                if os.path.exists(CSV_PATH):
                    df = read_csv(CSV_PATH)
                    if df is not None:
                        # Update session state with fresh data from disk
                        st.session_state.dataframe = df
                        
                        # Update file_data_list with CSV data for consistency
                        for i, file_data in enumerate(st.session_state.file_data_list):
                            # Find the matching row in the dataframe by full_path
                            matching_rows = df[df['full_path'] == file_data['full_path']]
                            if not matching_rows.empty:
                                row = matching_rows.iloc[0]
                                # Update the file data with the values from the CSV
                                for col in df.columns:
                                    if col in ['abstract', 'abstract_ita', 'abstract_eng', 'language', 'theme', 'format', 
                                             'geographic_area', 'keywords', 'original_magazine', 'original_magazine_no', 
                                             'original_author', 'original_title', 'magazine', 'magazine_no', 'author', 
                                             'title', 'preview_image_path', 'folder_images']:
                                        file_data[col] = row[col]
                                st.session_state.file_data_list[i] = file_data
                        
                        st.success("CSV data reloaded successfully.")
                    else:
                        st.error("Error reading CSV file.")
                else:
                    st.warning("No CSV file found to reload.")
                st.rerun()
            except Exception as e:
                st.error(f"Error reloading CSV data: {e}")
    
    # Initialize data display
    try:
        if not os.path.exists(CSV_PATH):
            # Create new DataFrame and CSV file if it doesn't exist
            df = create_dataframe(st.session_state.file_data_list)
            save_to_csv(df, CSV_PATH)
            st.session_state.dataframe = df
        elif 'dataframe' in st.session_state:
            # Use the stored DataFrame if available (updated by reconciliation)
            df = st.session_state.dataframe
        else:
            # Otherwise read from the CSV file
            df = read_csv(CSV_PATH)
            if df is None:
                # If there was an error reading the CSV, create a new one
                st.warning("Error reading CSV file. Creating a new one.")
                df = create_dataframe(st.session_state.file_data_list)
                save_to_csv(df, CSV_PATH)
            st.session_state.dataframe = df
        
        # Display a limited view of the DataFrame to reduce memory usage
        display_columns = ['title', 'magazine', 'magazine_no', 'author', 'theme', 'format', 'geographic_area']
        if all(col in df.columns for col in display_columns):
            st.dataframe(df[display_columns], use_container_width=True)
        else:
            # Fall back to showing all columns if the expected columns aren't available
            st.dataframe(df, use_container_width=True)
        
        # Download button for the CSV
        if os.path.exists(CSV_PATH) and os.path.getsize(CSV_PATH) > 0:
            with open(CSV_PATH, 'rb') as f:
                st.download_button(
                    label="Download CSV",
                    data=f,
                    file_name="extracted_data.csv",
                    mime="text/csv"
                )
        else:
            st.warning("CSV file is empty or not yet created. Process some files first.")
    except Exception as e:
        st.error(f"Error displaying data: {e}")
        # Try to recover by creating a new DataFrame
        try:
            df = create_dataframe(st.session_state.file_data_list)
            save_to_csv(df, CSV_PATH)
            st.session_state.dataframe = df
            # Use the same display columns pattern for recovery
            if 'display_columns' in locals() and all(col in df.columns for col in display_columns):
                st.dataframe(df[display_columns], use_container_width=True)
            else:
                st.dataframe(df, use_container_width=True)
            st.success("Successfully recovered data display.")
        except Exception as recovery_error:
            st.error(f"Unable to recover data display: {recovery_error}")
            
    # Tabs for file details
    tab_all, tab_phase1, tab_phase2, tab_complete = st.tabs(["All Files", "Phase 1 Needed", "Phase 2 Needed", "Completed"])
    
    with tab_all:
        # Add search and filtering options
        st.subheader("Filter Files")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            search_term = st.text_input("Search by title, author, or content", 
                                      help="Enter text to search in title, author, or content",
                                      placeholder="Enter search term...")
        
        with col2:
            filter_options = ["All"]
            # Add magazine options
            if 'dataframe' in st.session_state:
                magazines = st.session_state.dataframe['magazine'].unique().tolist()
                filter_options.extend([f"Magazine: {m}" for m in magazines if m and m != 'nan'])
            
            # Add filter options for missing/problematic data
            filter_options.extend([
                "Missing abstracts", 
                "Missing themes",
                "Missing authors", 
                "Missing titles",
                "Contains 'Unknown'",
                "Contains 'Not Specified'",
                "Contains 'nan'"
            ])
            
            filter_by = st.selectbox("Filter by", filter_options)
        
        with col3:
            sort_options = [
                "Title (A-Z)",
                "Title (Z-A)",
                "Magazine (A-Z)",
                "Magazine Issue (Newest)",
                "Magazine Issue (Oldest)",
                "Author (A-Z)",
                "Author (Z-A)"
            ]
            sort_by = st.selectbox("Sort by", sort_options)
        
        # Apply filters to the file list
        filtered_files = st.session_state.file_data_list.copy()
        
        # Apply search filter
        if search_term:
            search_term = search_term.lower()
            filtered_files = [
                f for f in filtered_files 
                if search_term in str(f.get('title', '')).lower() or 
                   search_term in str(f.get('author', '')).lower() or
                   search_term in str(f.get('abstract', '')).lower() or
                   search_term in str(f.get('text_content', '')).lower()
            ]
        
        # Apply dropdown filter
        if filter_by != "All":
            if filter_by.startswith("Magazine: "):
                magazine_name = filter_by.replace("Magazine: ", "")
                filtered_files = [f for f in filtered_files if f.get('magazine', '') == magazine_name]
            elif filter_by == "Missing abstracts":
                filtered_files = [
                    f for f in filtered_files 
                    if 'abstract' not in f or not f['abstract'] or 
                       f['abstract'] == 'nan' or f['abstract'] == 'Not Specified'
                ]
            elif filter_by == "Missing themes":
                filtered_files = [
                    f for f in filtered_files 
                    if 'theme' not in f or not f['theme'] or 
                       f['theme'] == 'nan' or f['theme'] == 'Not Specified'
                ]
            elif filter_by == "Missing authors":
                filtered_files = [
                    f for f in filtered_files 
                    if 'author' not in f or not f['author'] or 
                       f['author'] == 'nan' or f['author'] == 'Unknown'
                ]
            elif filter_by == "Missing titles":
                filtered_files = [
                    f for f in filtered_files 
                    if 'title' not in f or not f['title'] or 
                       f['title'] == 'nan' or f['title'] == 'Unknown'
                ]
            elif filter_by == "Contains 'Unknown'":
                filtered_files = [
                    f for f in filtered_files 
                    if any(str(v) == 'Unknown' for v in f.values())
                ]
            elif filter_by == "Contains 'Not Specified'":
                filtered_files = [
                    f for f in filtered_files 
                    if any(str(v) == 'Not Specified' for v in f.values())
                ]
            elif filter_by == "Contains 'nan'":
                filtered_files = [
                    f for f in filtered_files 
                    if any(str(v) == 'nan' for v in f.values())
                ]
        
        # Apply sorting
        if sort_by == "Title (A-Z)":
            filtered_files.sort(key=lambda x: str(x.get('title', '')).lower())
        elif sort_by == "Title (Z-A)":
            filtered_files.sort(key=lambda x: str(x.get('title', '')).lower(), reverse=True)
        elif sort_by == "Magazine (A-Z)":
            filtered_files.sort(key=lambda x: (str(x.get('magazine', '')).lower(), str(x.get('magazine_no', ''))))
        elif sort_by == "Magazine Issue (Newest)":
            filtered_files.sort(key=lambda x: (str(x.get('magazine', '')).lower(), str(x.get('magazine_no', ''))), reverse=True)
        elif sort_by == "Magazine Issue (Oldest)":
            filtered_files.sort(key=lambda x: (str(x.get('magazine', '')).lower(), str(x.get('magazine_no', ''))))
        elif sort_by == "Author (A-Z)":
            filtered_files.sort(key=lambda x: str(x.get('author', '')).lower())
        elif sort_by == "Author (Z-A)":
            filtered_files.sort(key=lambda x: str(x.get('author', '')).lower(), reverse=True)
        
        # Display filter results
        total_filtered = len(filtered_files)
        st.write(f"Showing {total_filtered} of {total_files} files")
        
        # Add pagination to limit the number of displayed files
        page_size = 10  # Show only 10 items per page
        total_pages = (total_filtered + page_size - 1) // page_size  # Ceiling division
        
        if total_pages > 1:
            page = st.number_input("Page", min_value=1, max_value=total_pages, value=1)
            st.write(f"Showing page {page} of {total_pages} ({total_filtered} total files)")
        else:
            page = 1
        
        # Calculate the slice of files to display
        start_idx = (page - 1) * page_size
        end_idx = min(start_idx + page_size, total_filtered)
        
        # Only display files for the current page
        for i, file_data in enumerate(filtered_files[start_idx:end_idx], start=start_idx):
            with st.expander(f"{file_data['title']} ({file_data['magazine']} {file_data['magazine_no']} - {file_data['author']})"):
                col1, col2 = st.columns([1, 3])
                
                with col1:
                    # Display preview image if available
                    if 'preview_image' in file_data and file_data['preview_image'] and file_data['preview_image'].startswith('data:'):
                        # Display base64 image (legacy support)
                        display_image(file_data['preview_image'])
                    elif 'preview_image_path' in file_data and file_data['preview_image_path'] and file_data['preview_image_path'] != "[embedded image in document]":
                        # Display image from file path
                        try:
                            img = Image.open(file_data['preview_image_path'])
                            img.thumbnail((200, 200), Image.LANCZOS)
                            st.image(img, width=200)
                        except Exception as e:
                            st.error(f"Error loading image: {e}")
                    # If no preview image but folder images exist, display the first folder image
                    elif 'folder_images' in file_data and file_data['folder_images']:
                        try:
                            # If it's already a list, use first item
                            if isinstance(file_data['folder_images'], list) and file_data['folder_images']:
                                # Load and display the first image
                                try:
                                    img = Image.open(file_data['folder_images'][0])
                                    img.thumbnail((200, 200), Image.LANCZOS)
                                    st.image(img, width=200)
                                    st.caption(f"1 of {len(file_data['folder_images'])} images")
                                except Exception as e:
                                    st.error(f"Error displaying folder image: {e}")
                            # If it's a string (from CSV), try to parse it
                            elif isinstance(file_data['folder_images'], str) and file_data['folder_images']:
                                folder_images = file_data['folder_images'].split('|')
                                if folder_images:
                                    try:
                                        img = Image.open(folder_images[0])
                                        img.thumbnail((200, 200), Image.LANCZOS)
                                        st.image(img, width=200)
                                        st.caption(f"1 of {len(folder_images)} images")
                                    except Exception as e:
                                        st.error(f"Error displaying folder image from CSV: {e}")
                        except Exception as e:
                            st.error(f"Error processing folder images: {e}")
                    else:
                        st.text("No images available")
                
                with col2:
                    # Display file path
                    st.text(f"File Path: {file_data['full_path']}")
                    
                    # Display metadata with comparison if reconciled
                    if 'original_magazine' in file_data and file_data['original_magazine'] != file_data['magazine']:
                        st.text(f"Magazine: {file_data['magazine']} (original: {file_data['original_magazine']})")
                    else:
                        st.text(f"Magazine: {file_data['magazine']}")
                    
                    if 'original_magazine_no' in file_data and file_data['original_magazine_no'] != file_data['magazine_no']:
                        st.text(f"Issue: {file_data['magazine_no']} (original: {file_data['original_magazine_no']})")
                    else:
                        st.text(f"Issue: {file_data['magazine_no']}")
                    
                    if 'original_author' in file_data and file_data['original_author'] != file_data['author']:
                        st.text(f"Author: {file_data['author']} (original: {file_data['original_author']})")
                    else:
                        st.text(f"Author: {file_data['author']}")
                    
                    if 'original_title' in file_data and file_data['original_title'] != file_data['title']:
                        st.text(f"Title: {file_data['title']} (original: {file_data['original_title']})")
                    else:
                        st.text(f"Title: {file_data['title']}")
                    
                    # Display language field
                    if 'language' in file_data:
                        st.text(f"Language: {file_data['language']}")
                    
                    # Display abstracts based on language if available
                    if 'language' in file_data and file_data['language'] == 'ITA' and 'abstract_ita' in file_data and file_data['abstract_ita']:
                        st.text("Abstract (ITA):")
                        st.write(file_data['abstract_ita'])
                        if 'abstract_eng' in file_data and file_data['abstract_eng']:
                            st.text("Abstract (ENG):")
                            st.write(file_data['abstract_eng'])
                    elif 'language' in file_data and file_data['language'] == 'ENG' and 'abstract_eng' in file_data and file_data['abstract_eng']:
                        st.text("Abstract (ENG):")
                        st.write(file_data['abstract_eng'])
                        if 'abstract_ita' in file_data and file_data['abstract_ita']:
                            st.text("Abstract (ITA):")
                            st.write(file_data['abstract_ita'])
                    elif 'abstract' in file_data:
                        # Fallback to original abstract
                        st.text("Abstract:")
                        st.write(file_data['abstract'])
                    
                    if 'theme' in file_data:
                        st.text(f"Theme: {file_data['theme']}")
                    
                    if 'format' in file_data:
                        st.text(f"Format: {file_data['format']}")
                    
                    if 'geographic_area' in file_data:
                        st.text(f"Geographic Area: {file_data['geographic_area']}")
                    
                    if 'keywords' in file_data:
                        st.text(f"Keywords: {file_data['keywords']}")
                
                # Check if API key is valid before allowing individual processing
                api_key_valid = st.session_state.get('api_key_valid', False) or validate_openai_api_key(config['openai']['api_key'])
                st.session_state.api_key_valid = api_key_valid
                
                if not api_key_valid:
                    st.error("Please enter a valid OpenAI API key before processing files")
                else:
                    # Display different action buttons based on processing status
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Phase 1 extraction button
                        phase1_needed = not all(k in file_data for k in ['abstract', 'theme', 'format', 'geographic_area', 'keywords'])
                        if phase1_needed:
                            if st.button(f"Phase 1: Extract Data", key=f"extract1_{i}"):
                                with st.spinner("Extracting data (Phase 1)..."):
                                    try:
                                        config = load_config()
                                        
                                        # Ensure API key is in config
                                        api_key = st.session_state.get('api_key', '') or os.environ.get('OPENAI_API_KEY', '')
                                        if api_key:
                                            config['openai']['api_key'] = api_key
                                            os.environ['OPENAI_API_KEY'] = api_key  # Also set in environment
                                        
                                        # Process the file (Phase 1)
                                        updated_file_data, df = process_file(file_data, df, config, reconcile=False)
                                        
                                        # Update the file_data_list
                                        st.session_state.file_data_list[i] = updated_file_data
                                        
                                        # Update the session state dataframe
                                        st.session_state.dataframe = df
                                        
                                        st.success(f"Successfully extracted data for '{updated_file_data['title']}'")
                                        
                                        # Rerun to refresh the UI
                                        st.rerun()
                                    except Exception as e:
                                        st.error(f"Error in Phase 1 extraction: {e}")
                        else:
                            # Retry Phase 1 button
                            if st.button(f"Retry Phase 1", key=f"retry1_{i}"):
                                with st.spinner("Retrying extraction (Phase 1)..."):
                                    try:
                                        config = load_config()
                                        
                                        # Ensure API key is in config
                                        api_key = st.session_state.get('api_key', '') or os.environ.get('OPENAI_API_KEY', '')
                                        if api_key:
                                            config['openai']['api_key'] = api_key
                                            os.environ['OPENAI_API_KEY'] = api_key  # Also set in environment
                                            
                                        # Process the file again (Phase 1)
                                        updated_file_data, df = process_file(file_data, df, config, reconcile=False)
                                        
                                        # Update the file_data_list
                                        st.session_state.file_data_list[i] = updated_file_data
                                        
                                        # Update the session state dataframe
                                        st.session_state.dataframe = df
                                        
                                        st.success(f"Successfully re-extracted data for '{updated_file_data['title']}'")
                                        
                                        # Rerun to refresh the UI
                                        st.rerun()
                                    except Exception as e:
                                        st.error(f"Error in Phase 1 re-extraction: {e}")
                    
                    with col2:
                        # Phase 2 reconciliation button
                        phase2_needed = all(k in file_data for k in ['abstract', 'theme', 'format', 'geographic_area', 'keywords'])
                        if phase2_needed:
                            button_label = "Phase 2: Reconcile Metadata"
                            if 'original_magazine' in file_data and file_data.get('original_magazine') != file_data.get('magazine'):
                                button_label = "Re-reconcile Metadata"
                            
                            if st.button(button_label, key=f"reconcile_{i}"):
                                with st.spinner("Reconciling metadata (Phase 2)..."):
                                    try:
                                        config = load_config()
                                        
                                        # Ensure API key is in config
                                        api_key = st.session_state.get('api_key', '') or os.environ.get('OPENAI_API_KEY', '')
                                        if api_key:
                                            config['openai']['api_key'] = api_key
                                            os.environ['OPENAI_API_KEY'] = api_key  # Also set in environment
                                            
                                        # Process the file with reconciliation (Phase 2)
                                        updated_file_data, df = process_file(file_data, df, config, reconcile=True)
                                        
                                        # Update the file_data_list
                                        st.session_state.file_data_list[i] = updated_file_data
                                        
                                        # Update the session state dataframe to ensure the display is refreshed
                                        st.session_state.dataframe = df
                                        
                                        # Check if any fields were corrected
                                        corrections = []
                                        if 'original_magazine' in updated_file_data and updated_file_data['original_magazine'] != updated_file_data['magazine']:
                                            corrections.append(f"Magazine: {updated_file_data['original_magazine']} → {updated_file_data['magazine']}")
                                        if 'original_magazine_no' in updated_file_data and updated_file_data['original_magazine_no'] != updated_file_data['magazine_no']:
                                            corrections.append(f"Issue: {updated_file_data['original_magazine_no']} → {updated_file_data['magazine_no']}")
                                        if 'original_author' in updated_file_data and updated_file_data['original_author'] != updated_file_data['author']:
                                            corrections.append(f"Author: {updated_file_data['original_author']} → {updated_file_data['author']}")
                                        if 'original_title' in updated_file_data and updated_file_data['original_title'] != updated_file_data['title']:
                                            corrections.append(f"Title: {updated_file_data['original_title']} → {updated_file_data['title']}")
                                        
                                        if corrections:
                                            st.success(f"Successfully reconciled metadata with corrections:\n" + "\n".join(corrections))
                                        else:
                                            st.success(f"Successfully reconciled metadata (no corrections needed)")
                                        
                                        # Rerun to refresh the UI
                                        st.rerun()
                                    except Exception as e:
                                        st.error(f"Error in Phase 2 reconciliation: {e}")
    
    with tab_phase1:
        phase1_files = [f for f in st.session_state.file_data_list 
                        if not all(k in f for k in ['abstract', 'theme', 'format', 'geographic_area', 'keywords'])]
        
        if phase1_files:
            st.text(f"{len(phase1_files)} files need Phase 1 extraction")
            
            # Add pagination for Phase 1 files
            p1_page_size = 20
            p1_total_pages = (len(phase1_files) + p1_page_size - 1) // p1_page_size
            
            if p1_total_pages > 1:
                p1_page = st.number_input("Phase 1 Page", min_value=1, max_value=p1_total_pages, value=1, key="p1_page")
                p1_start = (p1_page - 1) * p1_page_size
                p1_end = min(p1_start + p1_page_size, len(phase1_files))
                st.write(f"Showing {p1_start+1}-{p1_end} of {len(phase1_files)} files")
                display_files = phase1_files[p1_start:p1_end]
            else:
                display_files = phase1_files
                
            for file_data in display_files:
                st.text(f"{file_data['title']} ({file_data['magazine']} {file_data['magazine_no']} - {file_data['author']})")
        else:
            st.success("All files have completed Phase 1 extraction!")
    
    with tab_phase2:
        # Files that have completed Phase 1 but haven't been reconciled yet
        phase2_files = [f for f in st.session_state.file_data_list 
                       if all(k in f for k in ['abstract', 'theme', 'format', 'geographic_area', 'keywords']) and
                       ('original_magazine' not in f or 'original_title' not in f)]
        
        if phase2_files:
            st.text(f"{len(phase2_files)} files need Phase 2 reconciliation")
            
            # Add pagination for Phase 2 files
            p2_page_size = 20
            p2_total_pages = (len(phase2_files) + p2_page_size - 1) // p2_page_size
            
            if p2_total_pages > 1:
                p2_page = st.number_input("Phase 2 Page", min_value=1, max_value=p2_total_pages, value=1, key="p2_page")
                p2_start = (p2_page - 1) * p2_page_size
                p2_end = min(p2_start + p2_page_size, len(phase2_files))
                st.write(f"Showing {p2_start+1}-{p2_end} of {len(phase2_files)} files")
                display_files = phase2_files[p2_start:p2_end]
            else:
                display_files = phase2_files
                
            for file_data in display_files:
                st.text(f"{file_data['title']} ({file_data['magazine']} {file_data['magazine_no']} - {file_data['author']})")
        else:
            st.success("All extracted files have been reconciled!")
    
    with tab_complete:
        # Files that have completed both phases
        complete_files = [f for f in st.session_state.file_data_list 
                         if all(k in f for k in ['abstract', 'theme', 'format', 'geographic_area', 'keywords']) and
                         'original_magazine' in f and 'original_title' in f]
        
        if complete_files:
            st.text(f"{len(complete_files)} files have completed all processing phases")
            
            # Count files with corrections
            corrected_files = [f for f in complete_files 
                              if f['original_magazine'] != f['magazine'] or
                                 f['original_magazine_no'] != f['magazine_no'] or
                                 f['original_author'] != f['author'] or
                                 f['original_title'] != f['title']]
            
            if corrected_files:
                st.text(f"{len(corrected_files)} files had metadata corrections")
            
            # Add pagination for completed files
            c_page_size = 20
            c_total_pages = (len(complete_files) + c_page_size - 1) // c_page_size
            
            if c_total_pages > 1:
                c_page = st.number_input("Completed Page", min_value=1, max_value=c_total_pages, value=1, key="c_page")
                c_start = (c_page - 1) * c_page_size
                c_end = min(c_start + c_page_size, len(complete_files))
                st.write(f"Showing {c_start+1}-{c_end} of {len(complete_files)} files")
                display_files = complete_files[c_start:c_end]
            else:
                display_files = complete_files
            
            for file_data in display_files:
                has_corrections = (file_data['original_magazine'] != file_data['magazine'] or
                                  file_data['original_magazine_no'] != file_data['magazine_no'] or
                                  file_data['original_author'] != file_data['author'] or
                                  file_data['original_title'] != file_data['title'])
                
                if has_corrections:
                    st.text(f"✓ {file_data['title']} (corrected)")
                else:
                    st.text(f"✓ {file_data['title']}")
        else:
            st.warning("No files have completed all processing phases yet")
else:
    st.info("Select the Upload Files section in the sidebar to begin")

with tab_analysis:
    st.header("Magazine Data Analysis")
    
    st.markdown("""
    This section provides tools to analyze and explore the extracted magazine data.
    You can view statistics, validate data quality, and explore articles by magazine, issue, and theme.
    """)
    
    # Add a button to recalculate stats and clear cache
    col1, col2 = st.columns([1, 3])
    with col1:
        if st.button("🔄 Recalculate Stats", help="Refresh all statistics with the latest data"):
            # Clear the cached data to force recalculation
            st.cache_data.clear()
            st.success("Statistics recalculated with latest data!")
            st.rerun()
    with col2:
        if os.path.exists(CSV_PATH):
            last_modified = os.path.getmtime(CSV_PATH)
            st.info(f"Data last updated: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(last_modified))}")
    
    # Check if CSV file exists
    if not os.path.exists(CSV_PATH):
        st.warning("No CSV data found. Please extract some data first.")
    else:
        # Load the CSV data
        @st.cache_data
        def load_analysis_data():
            """Load data from CSV and perform basic preprocessing"""
            try:
                df = pd.read_csv(CSV_PATH)
                
                # Normalize magazine_no to numeric when possible
                df['magazine_no_numeric'] = pd.to_numeric(df['magazine_no'], errors='coerce')
                
                return df
            except Exception as e:
                st.error(f"Error loading CSV data: {str(e)}")
                return None
        
        # Load the data
        df = load_analysis_data()
        
        if df is None:
            st.error("Failed to load the CSV data. The file might be corrupt or empty.")
            st.stop()
        
        if len(df) == 0:
            st.warning("The CSV file exists but contains no data. Please extract some articles first.")
            st.stop()
            
        # Check for required columns
        required_cols = ['magazine', 'magazine_no', 'author', 'title']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            st.error(f"CSV data is missing required columns: {', '.join(missing_cols)}")
            st.info("Process files with extraction to create properly formatted data.")
            st.stop()
        
        # Initialize tabs for the Analysis section
        overview_tab, data_validation_tab, theme_tab, keyword_tab = st.tabs([
            "📈 Overview", 
            "🔍 Data Validation",
            "📊 Theme Analysis",
            "🔗 Keywords"
        ])
        
        # Display Overview tab
        with overview_tab:
            st.subheader("📈 Overview")
            
            st.markdown("""
            This tab provides a high-level summary of all the extracted magazine data, including 
            counts of articles, magazines, themes, and distribution charts.
            """)
            
            st.markdown("---")
            
            # Articles per magazine
            st.subheader("Articles per Magazine")
            fig = plot_magazine_distribution(df)
            st.plotly_chart(fig, use_container_width=True)
            
            # Articles per issue - using the new faceted chart
            st.subheader("Articles per Issue")
            fig = plot_issue_distribution(df)
            st.plotly_chart(fig, use_container_width=True)
            
            # Top authors
            st.subheader("Top 10 Authors")
            fig = plot_top_authors(df)
            st.plotly_chart(fig, use_container_width=True)
            
            # Format distribution chart - now with by-magazine breakdown
            st.subheader("Format Distribution")
            col1, col2 = st.columns(2)
            
            with col1:
                fig = plot_format_distribution(df)
                if fig is not None:
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Format information not available.")
            
            with col2:
                fig = plot_format_by_magazine(df)
                if fig is not None:
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Format information not available.")
            
            # Language distribution if available
            if 'language' in df.columns:
                st.subheader("Language Distribution")
                language_pie, language_stacked = plot_language_distribution(df)
                
                if language_pie is not None and language_stacked is not None:
                    col1, col2 = st.columns(2)
                    with col1:
                        st.plotly_chart(language_pie, use_container_width=True)
                    with col2:
                        st.plotly_chart(language_stacked, use_container_width=True)
                else:
                    st.info("Language distribution data not available.")
            else:
                st.info("Language data not available in the dataset.")
            
        with data_validation_tab:
            st.subheader("🔍 Data Validation")
            
            st.markdown("""
            Check data quality and validate metadata against expected values and ranges. This tab highlights 
            potential issues with magazine names, issue numbers, article counts, and missing data.
            """)
            
            st.markdown("---")
            
            # Define expected ranges and values
            expected_ranges = {
                'WE': list(range(34, 64)),  # 34 to 63 (expanded to include earlier issues)
                'Orizzonti': list(range(55, 65))  # 55 to 64
            }
            
            expected_counts = {
                'WE': 16,
                'Orizzonti': 12
            }
            
            # Run validation checks
            with st.spinner("Running validations..."):
                # Validate magazine names
                st.subheader("Magazine Names")
                valid_magazines = {'Orizzonti', 'WE'}
                invalid_magazines = df[~df['magazine'].isin(valid_magazines)]['magazine'].unique().tolist()
                
                if invalid_magazines:
                    st.error(f"Found {len(invalid_magazines)} invalid magazine names:")
                    for mag in invalid_magazines:
                        st.write(f"  - '{mag}'")
                else:
                    st.success("✅ All magazine names are valid (Orizzonti or WE)")
                
                # Validate issue numbers
                st.subheader("Issue Numbers")
                
                # Normalize magazine_no to integers
                df['magazine_no_norm'] = df['magazine_no'].apply(
                    lambda x: int(float(x)) if pd.notnull(x) and str(x).replace('.', '', 1).isdigit() else None
                )
                
                # Check for out of range issues
                out_of_range = []
                for magazine, expected_issues in expected_ranges.items():
                    magazine_df = df[df['magazine'] == magazine]
                    
                    for _, row in magazine_df.iterrows():
                        issue = row['magazine_no_norm']
                        # Skip 'Not Specified' or None values when checking ranges
                        if issue is not None and issue not in expected_issues and row['magazine_no'] != 'Not Specified':
                            out_of_range.append({
                                'magazine': magazine,
                                'issue': row['magazine_no'],
                                'normalized_issue': issue
                            })
                
                if out_of_range:
                    st.error(f"Found {len(out_of_range)} issues outside expected ranges:")
                    for item in out_of_range:
                        st.write(f"  - {item['magazine']} issue {item['issue']} (normalized: {item['normalized_issue']})")
                else:
                    st.success("✅ All magazine issues are within expected ranges")
                
                # Check for missing issues
                missing_issues = {}
                for magazine, expected_issues in expected_ranges.items():
                    magazine_df = df[df['magazine'] == magazine]
                    
                    if len(magazine_df) == 0:
                        missing_issues[magazine] = expected_issues
                        continue
                        
                    found_issues = set(magazine_df['magazine_no_norm'].dropna().unique())
                    magazine_missing = [i for i in expected_issues if i not in found_issues]
                    
                    if magazine_missing:
                        missing_issues[magazine] = magazine_missing
                
                missing_count = sum(len(issues) for issues in missing_issues.values())
                if missing_count > 0:
                    st.warning(f"Missing {missing_count} expected issues:")
                    for magazine, issues in missing_issues.items():
                        if issues:
                            st.write(f"  - {magazine}: missing issues {', '.join(map(str, issues))}")
                else:
                    st.success("✅ No missing issues detected")
                
                # Check article counts
                st.subheader("Article Counts")
                
                # Count articles per issue
                issue_counts = df.groupby(['magazine', 'magazine_no']).size()
                
                # Find issues with low article counts
                issues_with_low_counts = []
                threshold = 0.5  # 50% of expected
                
                for (magazine, issue), count in issue_counts.items():
                    if magazine in expected_counts:
                        expected = expected_counts[magazine]
                        if count < expected * threshold:
                            issues_with_low_counts.append({
                                'magazine': magazine,
                                'issue': issue,
                                'count': count,
                                'expected': expected,
                                'percentage': round(count / expected * 100, 1)
                            })
                
                if issues_with_low_counts:
                    st.warning(f"Found {len(issues_with_low_counts)} issues with fewer than 50% of expected articles:")
                    for item in issues_with_low_counts:
                        st.write(f"  - {item['magazine']} {item['issue']}: {item['count']} articles (only {item['percentage']}% of expected {item['expected']})")
                else:
                    st.success("✅ All issues have reasonable article counts")
                
                # Check for required fields
                st.subheader("Required Fields")
                
                required_fields = [
                    'author', 'title', 'magazine', 'magazine_no', 
                    'abstract', 'theme', 'format', 'geographic_area', 
                    'keywords', 'language'
                ]
                
                # Check for missing fields
                missing_fields = [field for field in required_fields if field not in df.columns]
                
                if missing_fields:
                    st.error(f"Missing required fields: {', '.join(missing_fields)}")
                else:
                    st.success("✅ All required fields exist in the dataset")
                
                # Check for null values
                null_counts = {}
                for field in required_fields:
                    if field in df.columns:
                        null_count = df[field].isnull().sum()
                        if null_count > 0:
                            null_counts[field] = null_count
                
                if null_counts:
                    st.warning("Found null values in these fields:")
                    for field, count in null_counts.items():
                        percentage = round(count / len(df) * 100, 1)
                        st.write(f"  - {field}: {count} nulls ({percentage}%)")
                else:
                    st.success("✅ No null values found in required fields")
                    
        with theme_tab:
            st.subheader("📊 Theme Analysis")
            
            st.markdown("""
            Analyze themes and their distribution across magazines and issues. View overall theme distribution,
            themes by specific magazine, and track theme trends across different issues.
            """)
            
            st.markdown("---")
            
            if 'theme' not in df.columns:
                st.info("Theme information is not available. Process files with Phase 1 extraction first.")
                st.stop()
            
            # Overall Theme Distribution
            st.subheader("Overall Theme Distribution")
            fig = plot_theme_distribution(df, top_n=15, orientation='h')  
            st.plotly_chart(fig, use_container_width=True)
            
            # Themes by Magazine
            st.subheader("Theme Distribution by Magazine")
            
            # Create a radio button to select magazine
            magazine = st.radio("Select Magazine", sorted(df['magazine'].unique()))
            
            # Filter to selected magazine
            magazine_df = df[df['magazine'] == magazine]
            
            if len(magazine_df) == 0:
                st.warning(f"No articles found for magazine: {magazine}")
                st.stop()
            
            # Use the pie chart from our visualization module
            fig = plot_theme_pie(magazine_df, title=f"Theme Distribution in {magazine}")
            st.plotly_chart(fig, use_container_width=True)
            
            # Theme trends across issues - now using Plotly
            st.subheader("Theme Trends Across Issues")
            
            # Get top 5 themes for this magazine
            magazine_df = magazine_df.copy()  # Create a true copy to avoid the warning
            magazine_df.loc[:, 'theme_str'] = magazine_df['theme'].astype(str)
            top_themes = magazine_df['theme_str'].value_counts().head(5).index.tolist()
            
            # Let user select themes to display
            selected_themes = st.multiselect(
                "Select Themes to Display",
                sorted(magazine_df['theme'].astype(str).unique()),
                default=top_themes[:3] if top_themes else None
            )
            
            if selected_themes:
                # Use our new Plotly-based theme trends function
                fig = plot_theme_trends_plotly(df, magazine, selected_themes)
                if fig is not None:
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("Not enough data to create theme trends chart for the selected themes.")
            else:
                st.info("Please select at least one theme to display.")

        with keyword_tab:
            st.subheader("🔗 Keyword Network Analysis")
            
            st.markdown("""
            This visualization shows connections between keywords that appear together in articles.
            Each node represents a keyword, and links indicate keywords that appear in the same article.
            Larger nodes have more connections, and keywords that are also themes are highlighted in orange.
            
            Use the controls below to filter and customize the visualization.
            """)
            
            # Add filters and controls in a sidebar-like container
            with st.container():
                col1, col2, col3 = st.columns([1, 1, 1])
                
                with col1:
                    # Corpus selection
                    st.markdown("#### 📄 Content Filtering")
                    
                    # Add a filter by magazine
                    magazine_filter = st.multiselect(
                        "Filter by Magazine", 
                        options=df['magazine'].unique(),
                        default=list(df['magazine'].unique()),
                        help="Select which magazines to include in the analysis"
                    )
                    
                    # Add a filter by theme
                    theme_options = df['theme'].dropna().unique() if 'theme' in df.columns else []
                    theme_filter = st.multiselect(
                        "Filter by Theme",
                        options=theme_options,
                        default=[],
                        help="Select themes to filter the data (leave empty to include all)"
                    )
                
                with col2:
                    # Controls for visualization
                    st.markdown("#### 🎮 Visualization Controls")
                    
                    # Add a minimum link weight filter
                    min_weight = st.slider(
                        "Minimum co-occurrence", 
                        min_value=1, 
                        max_value=5, 
                        value=1,
                        help="Filter links by minimum number of times keywords appear together"
                    )
                    
                    # Add a maximum nodes filter to improve performance
                    max_nodes = st.slider(
                        "Maximum nodes", 
                        min_value=20, 
                        max_value=200, 
                        value=100,
                        help="Limit the number of nodes to improve performance"
                    )
                    
                    # Add a filter for minimum keyword frequency
                    min_freq = st.slider(
                        "Minimum keyword frequency", 
                        min_value=1, 
                        max_value=10, 
                        value=1,
                        help="Only include keywords that appear in at least this many articles"
                    )
                
                with col3:
                    # Analysis settings
                    st.markdown("#### 📊 Analysis Settings")
                    
                    # Split keywords option
                    split_keywords = st.checkbox(
                        "Split compound keywords", 
                        value=False,
                        help="Split keywords like 'renewable energy' into 'renewable' and 'energy'"
                    )
                    
                    # Stopwords option
                    use_stopwords = st.checkbox(
                        "Filter common stopwords", 
                        value=True,
                        help="Remove common words like 'and', 'the', etc."
                    )
                    
                    # Language selector for stopwords
                    stopword_language = st.selectbox(
                        "Stopword language",
                        options=["italian", "english", "both"],
                        index=2,
                        help="Language for stopwords filtering",
                        disabled=not use_stopwords
                    )
            
            # Apply the filters to the data
            filtered_df = df.copy()
            
            # Apply magazine filter
            if magazine_filter and len(magazine_filter) < len(df['magazine'].unique()):
                filtered_df = filtered_df[filtered_df['magazine'].isin(magazine_filter)]
            
            # Apply theme filter
            if theme_filter and 'theme' in filtered_df.columns:
                filtered_df = filtered_df[filtered_df['theme'].isin(theme_filter)]
            
            # Show info about filtered dataset
            st.caption(f"Analyzing {len(filtered_df)} articles from {len(filtered_df['magazine'].unique())} magazines")
            
            # Get the keywords data
            if 'keywords' not in filtered_df.columns:
                st.warning("No keywords data available. Please run Phase 1 extraction on your files.")
            else:
                # Process keywords based on options
                if split_keywords:
                    # Split compound keywords
                    processed_keywords = []
                    for idx, row in filtered_df.iterrows():
                        if pd.isna(row.get('keywords')):
                            continue
                            
                        keywords = row['keywords'].split(',')
                        expanded_keywords = []
                        
                        for kw in keywords:
                            kw = kw.strip()
                            if ' ' in kw:
                                # Split multi-word keywords
                                parts = [p.strip() for p in kw.split(' ') if p.strip()]
                                expanded_keywords.extend(parts)
                            if kw:  # Also keep the original keyword
                                expanded_keywords.append(kw)
                        
                        # Create a copy of the row
                        row_copy = row.copy()
                        row_copy['keywords'] = ','.join(expanded_keywords)
                        processed_keywords.append(row_copy)
                        
                    if processed_keywords:
                        # Create a new dataframe with the processed keywords
                        temp_df = pd.DataFrame(processed_keywords)
                        # Build network from the processed data
                        network_data = build_keyword_network(temp_df)
                    else:
                        network_data = {"nodes": [], "links": []}
                else:
                    # Use original keywords
                    network_data = build_keyword_network(filtered_df)
                
                # Apply stopwords filtering if enabled
                if use_stopwords:
                    try:
                        # Try to import NLTK
                        import nltk
                        from nltk.corpus import stopwords
                        
                        # Download stopwords if not already downloaded
                        nltk.download('stopwords', quiet=True)
                        
                        # Get stopwords based on selected language
                        stopword_set = set()
                        if stopword_language in ["italian", "both"]:
                            stopword_set.update(stopwords.words('italian'))
                        if stopword_language in ["english", "both"]:
                            stopword_set.update(stopwords.words('english'))
                        
                        # Filter nodes to remove stopwords
                        network_data["nodes"] = [
                            node for node in network_data["nodes"] 
                            if node["id"].lower() not in stopword_set
                        ]
                        
                        # Filter links to remove stopwords
                        network_data["links"] = [
                            link for link in network_data["links"]
                            if (link["source"] not in stopword_set and 
                                link["target"] not in stopword_set)
                        ]
                    except (ImportError, ModuleNotFoundError):
                        st.warning("NLTK is not installed. Stopwords filtering is disabled. To enable, install with: pip install nltk")
                        use_stopwords = False
                    except Exception as e:
                        st.warning(f"Could not filter stopwords: {str(e)}")
                        use_stopwords = False
                
                # Filter links by weight
                if min_weight > 1:
                    network_data["links"] = [link for link in network_data["links"] if link["weight"] >= min_weight]
                
                # Count node connections and frequencies after filtering links
                node_connections = {}
                node_weights = {}
                
                for link in network_data["links"]:
                    # Count connections
                    node_connections[link["source"]] = node_connections.get(link["source"], 0) + 1
                    node_connections[link["target"]] = node_connections.get(link["target"], 0) + 1
                    
                    # Sum weights
                    node_weights[link["source"]] = node_weights.get(link["source"], 0) + link["weight"]
                    node_weights[link["target"]] = node_weights.get(link["target"], 0) + link["weight"]
                
                # Filter by minimum frequency
                if min_freq > 1:
                    # Keep only nodes that appear in at least min_freq articles
                    freq_filtered_nodes = {node_id for node_id, weight in node_weights.items() if weight >= min_freq}
                    network_data["nodes"] = [node for node in network_data["nodes"] if node["id"] in freq_filtered_nodes]
                    network_data["links"] = [
                        link for link in network_data["links"] 
                        if link["source"] in freq_filtered_nodes and link["target"] in freq_filtered_nodes
                    ]
                
                # Update the connections after filtering
                node_connections = {}
                for link in network_data["links"]:
                    node_connections[link["source"]] = node_connections.get(link["source"], 0) + 1
                    node_connections[link["target"]] = node_connections.get(link["target"], 0) + 1
                
                # Filter nodes by connection count and limit to max_nodes
                connected_nodes = sorted(node_connections.items(), key=lambda x: x[1], reverse=True)
                if len(connected_nodes) > max_nodes:
                    connected_nodes = connected_nodes[:max_nodes]
                    
                # Keep only the top nodes
                top_node_ids = {node[0] for node in connected_nodes}
                network_data["nodes"] = [node for node in network_data["nodes"] if node["id"] in top_node_ids]
                network_data["links"] = [
                    link for link in network_data["links"] 
                    if link["source"] in top_node_ids and link["target"] in top_node_ids
                ]
                
                # Show network stats before rendering
                st.markdown("---")
                
                # Add some interesting network metrics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Keywords", len(network_data['nodes']))
                
                with col2:
                    st.metric("Connections", len(network_data['links']))
                
                with col3:
                    if len(network_data['nodes']) > 0:
                        density = len(network_data['links']) / (len(network_data['nodes']) * (len(network_data['nodes'])-1)/2) if len(network_data['nodes']) > 1 else 0
                        st.metric("Network Density", f"{density:.2%}")
                    else:
                        st.metric("Network Density", "N/A")
                
                with col4:
                    if len(connected_nodes) > 0:
                        avg_connections = sum(count for _, count in connected_nodes) / len(connected_nodes)
                        st.metric("Avg. Connections", f"{avg_connections:.1f}")
                    else:
                        st.metric("Avg. Connections", "N/A")
                
                # Render the network
                with st.spinner("Generating network visualization..."):
                    render_keyword_network(network_data)

def create_lightweight_file_data(file_data_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Create a lightweight copy of file_data_list for session state storage.
    Removes large text content and base64 images.
    
    Args:
        file_data_list: The original file data list
        
    Returns:
        A lightweight copy with large data removed
    """
    if not file_data_list:
        return []
    
    lightweight_list = []
    
    for file_data in file_data_list:
        # Create a shallow copy of the file data
        light_data = {}
        
        # Copy only essential metadata, skip large content
        for key, value in file_data.items():
            # Skip base64 images
            if key == 'preview_image' and isinstance(value, str) and value.startswith('data:image'):
                continue
                
            # Skip full text content, but keep a small preview
            elif key == 'text_content' and isinstance(value, str):
                # Store length information
                light_data['text_content_length'] = len(value)
                # Keep just a small preview
                preview_length = min(len(value), 100)
                light_data['text_content_preview'] = value[:preview_length] + ('...' if preview_length < len(value) else '')
                
            # Keep path to preview image
            elif key == 'preview_image_path':
                light_data[key] = value
                
            # Keep all metadata
            elif key in ['filename', 'full_path', 'title', 'author', 'abstract', 'theme', 
                        'format', 'geographic_area', 'keywords', 'magazine', 'magazine_no', 
                        'language', 'id', 'folder_images']:
                light_data[key] = value
        
        lightweight_list.append(light_data)
    
    return lightweight_list

def display_file_details(file_data):
    """Display details of a selected file."""
    try:
        # File preview and basic details
        col1, col2 = st.columns([1, 3])
        
        with col1:
            # Display preview image if available
            if 'preview_image_path' in file_data and file_data['preview_image_path']:
                # Try to display from image path first (new method)
                display_image(file_data['preview_image_path'])
            elif 'preview_image' in file_data and file_data['preview_image']:
                # Fall back to base64 image (legacy method)
                display_image(file_data['preview_image'])
            else:
                st.info("No preview image available")
        
        with col2:
            # Display basic file info
            if 'filename' in file_data:
                st.subheader(file_data['filename'])
            
            # Show full path (useful for debugging)
            if 'full_path' in file_data:
                st.text(f"Path: {file_data['full_path']}")
                
            # Display extracted fields
            fields_to_display = [
                ('title', 'Title'),
                ('author', 'Author'),
                ('magazine', 'Magazine'),
                ('magazine_no', 'Magazine Number'),
                ('language', 'Language')
            ]
            
            for field_key, field_label in fields_to_display:
                if field_key in file_data and file_data[field_key]:
                    st.text(f"{field_label}: {file_data[field_key]}")
            
        # Expandable sections for more details
        with st.expander("Abstract"):
            if 'abstract' in file_data and file_data['abstract']:
                st.write(file_data['abstract'])
            else:
                st.info("No abstract available")
                
        with st.expander("Metadata"):
            # Create two columns for metadata
            meta_col1, meta_col2 = st.columns(2)
            
            with meta_col1:
                if 'theme' in file_data:
                    st.write(f"**Theme:** {file_data['theme']}")
                if 'format' in file_data:
                    st.write(f"**Format:** {file_data['format']}")
                if 'geographic_area' in file_data:
                    st.write(f"**Geographic Area:** {file_data['geographic_area']}")
                    
            with meta_col2:
                if 'keywords' in file_data:
                    st.write("**Keywords:**")
                    keywords = file_data['keywords'].split(',')
                    for keyword in keywords:
                        st.write(f"- {keyword.strip()}")
        
        # Display folder images if available
        if 'folder_images' in file_data and file_data['folder_images']:
            with st.expander("Additional Images"):
                for img_path in file_data['folder_images']:
                    st.text(os.path.basename(img_path))
                    display_image(img_path)
        
        # Show reconciliation data if available
        if all(k in file_data for k in ['original_magazine', 'original_title']):
            with st.expander("Original vs. Reconciled Metadata"):
                cols = st.columns(2)
                with cols[0]:
                    st.write("**Original Values:**")
                    st.write(f"Magazine: {file_data.get('original_magazine', 'N/A')}")
                    st.write(f"Issue: {file_data.get('original_magazine_no', 'N/A')}")
                    st.write(f"Author: {file_data.get('original_author', 'N/A')}")
                    st.write(f"Title: {file_data.get('original_title', 'N/A')}")
                    
                with cols[1]:
                    st.write("**Reconciled Values:**")
                    st.write(f"Magazine: {file_data.get('magazine', 'N/A')}")
                    st.write(f"Issue: {file_data.get('magazine_no', 'N/A')}")
                    st.write(f"Author: {file_data.get('author', 'N/A')}")
                    st.write(f"Title: {file_data.get('title', 'N/A')}")
            
    except Exception as e:
        st.error(f"Error displaying file details: {e}")
