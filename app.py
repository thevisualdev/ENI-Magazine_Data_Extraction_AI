import os
import streamlit as st
import pandas as pd
import time
import yaml
import tempfile
from PIL import Image
import io
import base64
import threading
import queue
from typing import List, Dict, Any, Optional, Tuple
import dotenv

# Load environment variables from .env file
dotenv.load_dotenv()

# Import custom modules
from file_manager import load_config, process_zip_file, process_directory
from docx_parser import extract_docx_content, save_preview_image
from ai_extractor import extract_fields_from_text, batch_extract_fields, setup_openai_api
from ai_reconciler import reconcile_metadata, batch_reconcile_metadata
from csv_manager import create_dataframe, save_to_csv, update_dataframe, read_csv

# Set page config
st.set_page_config(
    page_title="ENI Magazine Data Extraction",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

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

def display_image(image_base64):
    """Display an image from base64 string in Streamlit."""
    if image_base64:
        try:
            # Extract the image data
            image_data = image_base64.split(',')[1]
            
            # Decode base64 data
            image_bytes = base64.b64decode(image_data)
            
            # Open image using PIL
            image = Image.open(io.BytesIO(image_bytes))
            
            # Display the image
            st.image(image, width=200)
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
            text_content, preview_image = extract_docx_content(file_data_copy['full_path'])
            
            file_data_copy['text_content'] = text_content
            file_data_copy['preview_image'] = preview_image
        
        # Phase 1: Extract fields using OpenAI API (if not already done or if explicitly re-running Phase 1)
        phase1_needed = not all(k in file_data_copy for k in ['abstract', 'theme', 'format', 'geographic_area', 'keywords'])
        if phase1_needed or (not reconcile):
            # If re-running Phase 1 after Phase 2, preserve original metadata fields
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
                
                # Restore original fields
                file_data_copy.update(original_fields)
            else:
                # Normal Phase 1 extraction
                # Get the file path
                file_path = file_data_copy.get('full_path', '')
                
                # Extract new fields
                extracted_fields = extract_fields_from_text(file_path, file_data_copy['text_content'], config)
                
                # Update file data with extracted fields
                file_data_copy.update(extracted_fields)
        
        # Phase 2: Reconcile metadata if requested
        if reconcile:
            # Ensure we have Phase 1 data before attempting Phase 2
            if not all(k in file_data_copy for k in ['abstract', 'theme', 'format', 'geographic_area', 'keywords']):
                st.warning("Cannot reconcile metadata without Phase 1 extraction. Running Phase 1 first.")
                # Extract fields
                extracted_fields = extract_fields_from_text(file_data_copy['text_content'], config)
                file_data_copy.update(extracted_fields)
            
            # Store original values before reconciliation
            if 'original_magazine' not in file_data_copy:
                file_data_copy['original_magazine'] = file_data_copy.get('magazine', '')
            if 'original_magazine_no' not in file_data_copy:
                file_data_copy['original_magazine_no'] = file_data_copy.get('magazine_no', '')
            if 'original_author' not in file_data_copy:
                file_data_copy['original_author'] = file_data_copy.get('author', '')
            if 'original_title' not in file_data_copy:
                file_data_copy['original_title'] = file_data_copy.get('title', '')
            
            # Reconcile metadata with AI
            reconciled_data = reconcile_metadata(file_data_copy, config)
            
            # Update file data with reconciled fields
            file_data_copy.update(reconciled_data)
        
        # Update the DataFrame
        df = update_dataframe(df, file_data_copy)
        
        # Save the updated DataFrame to CSV
        save_to_csv(df, CSV_PATH)
        
        # Reload the CSV to ensure we have the latest data
        df_refreshed = read_csv(CSV_PATH)
        if df_refreshed is not None:
            df = df_refreshed
        
        return file_data_copy, df
    except Exception as e:
        st.error(f"Error processing file: {e}")
        # Return the original file_data and DataFrame if there's an error
        return file_data, df

def batch_processor():
    """Background thread for batch processing files."""
    global processing_queue, stop_processing, batch_success_count, batch_error_count
    global current_file_name, current_phase, processed_count, progress_percentage
    
    # Reset counters at start
    batch_success_count = 0
    batch_error_count = 0
    processed_count = 0
    progress_percentage = 0.0
    
    # Get file_data_list and other variables from session state once at the beginning
    try:
        # Make a local copy of data we need from session state
        file_data_list = list(st.session_state.file_data_list) 
        total_queue_size = st.session_state.total_queue_size
        
        # Get API key from session state - this is the most reliable source
        api_key = st.session_state.get('api_key', '')
        print(f"API key from session state: {'Found' if api_key else 'Not found'}")
    except Exception as e:
        print(f"Error accessing session state in batch processor startup: {e}")
        file_data_list = []
        total_queue_size = processing_queue.qsize()
        api_key = ''
    
    # Try to get API key from environment if not in session state
    if not api_key:
        api_key = os.environ.get('OPENAI_API_KEY', '')
        print(f"API key from environment: {'Found' if api_key else 'Not found'}")
    
    config = load_config()
    
    # Ensure the API key is in the config for the batch processing
    if api_key:
        config['openai']['api_key'] = api_key
        # Set it in the environment as well for maximum compatibility
        os.environ['OPENAI_API_KEY'] = api_key
        print(f"API key configured for batch processing - added to both config and environment")
    else:
        # Final fallback - check if config file has a key
        api_key = config['openai'].get('api_key', '')
        if api_key:
            os.environ['OPENAI_API_KEY'] = api_key
            print(f"Using API key from config file")
        else:
            print(f"WARNING: No API key found for batch processing!")
            
    # Double-check we have an API key - if not, we can't process files 
    if not api_key:
        print("CRITICAL ERROR: No API key available. Batch processing cannot continue.")
        stop_processing.set()
        return
    
    # Load existing data or create new DataFrame
    df = read_csv(CSV_PATH) if os.path.exists(CSV_PATH) else pd.DataFrame()
    if df is None or df.empty:
        # If there was an error reading the CSV or it's empty, create a new DataFrame
        try:
            df = create_dataframe(file_data_list)
        except Exception as df_error:
            print(f"Error creating DataFrame: {df_error}")
            df = pd.DataFrame()
        save_to_csv(df, CSV_PATH)
    
    # Track successful and failed files
    success_count = 0
    error_count = 0
    
    # Get total queue size
    if total_queue_size == 0:
        # If total_queue_size is not set, use queue size (less accurate but better than nothing)
        total_queue_size = processing_queue.qsize() 
    
    print(f"Starting batch processor with {total_queue_size} files in queue")
    
    while (not stop_processing.is_set() and not processing_queue.empty()):
        try:
            # Get the next file to process (with a timeout)
            file_data, phase = processing_queue.get(timeout=1)
            
            if stop_processing.is_set():
                processing_queue.task_done()
                break
                
            # Update thread-safe variables for UI display
            current_file_name = file_data.get('title', 'Unknown')
            current_phase = f"Phase {phase}"
            
            print(f"Processing file: {current_file_name} (Phase {phase})")
            
            try:
                # Process the file
                reconcile = (phase == 2)
                updated_file_data, df = process_file(file_data, df, config, reconcile=reconcile)
                
                # Store the updated file_data back to our local copy
                for i, item in enumerate(file_data_list):
                    if item.get('full_path') == updated_file_data.get('full_path'):
                        file_data_list[i] = updated_file_data
                        break
                
                # Force reload from the CSV to ensure consistency
                df_refreshed = read_csv(CSV_PATH)
                if df_refreshed is not None:
                    df = df_refreshed
                
                # Mark as success
                success_count += 1
                batch_success_count = success_count  # Update thread-safe counter
                print(f"Successfully processed file: {current_file_name} (Phase {phase})")
                
            except Exception as process_error:
                # Count the error but continue processing
                error_count += 1
                batch_error_count = error_count  # Update thread-safe counter
                print(f"Error processing file {current_file_name}: {process_error}")
            
            # Mark this task as done
            processing_queue.task_done()
            
            # Increment processed count
            processed_count += 1
            
            # Update progress percentage more precisely
            if total_queue_size > 0:
                progress_pct = min(1.0, processed_count / total_queue_size)
                progress_percentage = progress_pct
            
        except queue.Empty:
            # No items in the queue, just continue
            continue
        except Exception as e:
            print(f"Error in batch processor loop: {e}")
            
            # Count as error but continue processing other files
            error_count += 1
            batch_error_count = error_count  # Update thread-safe counter
            
            # Make sure we mark the task as done to avoid blocking
            try:
                processing_queue.task_done()
            except:
                pass
    
    # Ensure final state is saved
    # Force one last reload of the CSV at the end of batch processing
    try:
        df_final = read_csv(CSV_PATH)
        if df_final is not None:
            # Can't directly update session state from thread
            df = df_final
    except Exception as e:
        print(f"Error loading final CSV: {e}")
    
    # Provide a summary of processing
    if success_count > 0 or error_count > 0:
        print(f"Batch processing complete: {success_count} files processed successfully, {error_count} errors")
        if error_count > 0:
            print("Check the console logs above for details about errors")
    else:
        print("Batch processing completed with no files processed")
    
    # Signal that processing is complete
    # We can't update session state directly, but we keep
    # thread-safe variables updated for the main thread to read
    print("Batch processing finished")
    stop_processing.set()  # Signal that we're done

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
                            if col in ['abstract', 'theme', 'format', 'geographic_area', 'keywords',
                                      'original_magazine', 'original_magazine_no', 'original_author', 
                                      'original_title', 'magazine', 'magazine_no', 'author', 'title']:
                                file_data[col] = row[col]
                        st.session_state.file_data_list[i] = file_data
                        
                # Set dataframe in session state        
                st.session_state.dataframe = df

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
st.title("ENI Magazine Data Extraction")
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
    
    st.markdown("---")
    
    # File/Folder Upload Section
    st.header("Upload Files")
    
    # Add tabs for different upload methods
    upload_tab1, upload_tab2 = st.tabs(["Upload ZIP", "Select Directory"])
    
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
        Enter the full path to a local directory containing the ENI magazine folder structure.
        
        **Expected structure**: 
        - Root folder contains magazine folders (e.g., `Orizzonti_55`)
        - Each magazine folder contains author folders
        - Each author folder contains DOCX files
        
        The application will try to handle exceptions in the folder structure.
        """)
        
        directory_path = st.text_input(
            "Directory path",
            help="Enter the full path to the folder containing the magazine structure"
        )
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            # Add a button to process the directory
            if directory_path and st.button("Process Directory"):
                with st.spinner("Processing directory..."):
                    try:
                        # Process the directory
                        config = load_config()
                        file_data_list = process_directory(directory_path, config)
                        st.session_state.file_data_list = file_data_list
                        st.success(f"Processed directory. Found {len(file_data_list)} DOCX files.")
                    except Exception as e:
                        st.error(f"Error processing directory: {e}")
        
        with col2:
            # Add a button to show common paths as examples
            if st.button("Show Examples"):
                st.info("""
                Example paths:
                - macOS: /Users/username/Documents/ENI_Magazines
                - Windows: C:\\Users\\username\\Documents\\ENI_Magazines
                - Linux: /home/username/ENI_Magazines
                """)
    
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
        
        if not api_key_valid:
            st.error("Please enter a valid OpenAI API key before processing files")
        elif st.session_state.is_processing:
            if st.button("Stop Processing"):
                stop_processing.set()  # Signal the thread to stop
                st.warning("Stopping after current file completes...")
        else:
            if st.button("Start Batch Processing"):
                # Reset the stop flag
                stop_processing.clear()
                
                # Clear any existing queue
                while not processing_queue.empty():
                    try:
                        processing_queue.get_nowait()
                        processing_queue.task_done()
                    except:
                        pass
                
                # Start batch processing
                st.session_state.is_processing = True
                st.session_state.processed_count = 0
                st.session_state.progress_percentage = 0.0
                
                selected_phase = st.session_state.processing_phase
                
                # Count files that need processing
                phase1_count = 0
                phase2_count = 0
                
                # Put all unprocessed files in the queue based on selected phase
                for file_data in st.session_state.file_data_list:
                    if selected_phase == 0 or selected_phase == 1:  # Phase 1 or Both
                        if not all(k in file_data for k in ['abstract', 'theme', 'format', 'geographic_area', 'keywords']):
                            processing_queue.put((file_data, 1))  # Phase 1
                            phase1_count += 1
                    
                    if selected_phase == 0 or selected_phase == 2:  # Phase 2 or Both
                        # For Phase 2, we need files that have completed Phase 1
                        if all(k in file_data for k in ['abstract', 'theme', 'format', 'geographic_area', 'keywords']):
                            processing_queue.put((file_data, 2))  # Phase 2
                            phase2_count += 1
                
                total_queue_size = phase1_count + phase2_count
                st.session_state.total_queue_size = total_queue_size
                
                if total_queue_size == 0:
                    st.warning("No files need processing for the selected phase. Try selecting a different phase.")
                    st.session_state.is_processing = False
                else:
                    # Show what's being processed
                    if selected_phase == 0:
                        st.info(f"Processing {phase1_count} files in Phase 1 and {phase2_count} files in Phase 2.")
                    elif selected_phase == 1:
                        st.info(f"Processing {phase1_count} files in Phase 1.")
                    elif selected_phase == 2:
                        st.info(f"Processing {phase2_count} files in Phase 2.")
                    
                    # Start the background thread
                    processing_thread = threading.Thread(target=batch_processor)
                    processing_thread.daemon = True
                    processing_thread.start()
        
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
                
                progress_bar = st.progress(progress_pct)
            
            with col2:
                st.text(f"{processed_files} of {total_queue_size} files")
            
            # Show currently processing file with more details
            if st.session_state.current_processing_file:
                status_col1, status_col2 = st.columns([1, 1])
                with status_col1:
                    st.text(f"Currently processing: {st.session_state.current_processing_file}")
                with status_col2:
                    if st.session_state.current_processing_phase:
                        st.text(f"Current phase: {st.session_state.current_processing_phase}")
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
    
    # Add button to reset CSV data
    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        if st.button("Reset CSV Data"):
            try:
                # Remove the CSV file
                if os.path.exists(CSV_PATH):
                    os.remove(CSV_PATH)
                    
                # Clear the dataframe from session state    
                if 'dataframe' in st.session_state:
                    del st.session_state['dataframe']
                    
                # Create a new DataFrame and save it
                df = create_dataframe(st.session_state.file_data_list)
                save_to_csv(df, CSV_PATH)
                st.session_state.dataframe = df
                
                st.success("CSV data has been reset successfully.")
                st.rerun()
            except Exception as e:
                st.error(f"Error resetting CSV data: {e}")
    
    with col2:
        if st.button("Reload CSV Data"):
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
                                    if col in ['abstract', 'theme', 'format', 'geographic_area', 'keywords',
                                              'original_magazine', 'original_magazine_no', 'original_author', 
                                              'original_title', 'magazine', 'magazine_no', 'author', 'title']:
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
        
        # Display the DataFrame
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
            st.dataframe(df, use_container_width=True)
            st.success("Successfully recovered data display.")
        except Exception as recovery_error:
            st.error(f"Unable to recover data display: {recovery_error}")
            
    # Tabs for file details
    tab_all, tab_phase1, tab_phase2, tab_complete = st.tabs(["All Files", "Phase 1 Needed", "Phase 2 Needed", "Completed"])
    
    with tab_all:
        for i, file_data in enumerate(st.session_state.file_data_list):
            with st.expander(f"{file_data['title']} ({file_data['magazine']} {file_data['magazine_no']} - {file_data['author']})"):
                col1, col2 = st.columns([1, 3])
                
                with col1:
                    # Display preview image if available
                    if 'preview_image' in file_data and file_data['preview_image']:
                        display_image(file_data['preview_image'])
                    else:
                        st.text("No preview image")
                
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
                    
                    # Display extracted fields if available
                    if 'abstract' in file_data:
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
            for file_data in phase1_files:
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
            for file_data in phase2_files:
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
            
            for file_data in complete_files:
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