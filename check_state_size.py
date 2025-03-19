#!/usr/bin/env python3
"""
Check Streamlit Session State Size
This script checks the size of the session state files and cleans them if necessary.
"""

import os
import glob
import subprocess
import sys

def main():
    # Look for session state files
    session_files = glob.glob("./.streamlit/session_state_*.json")
    temp_state_file = "./.streamlit/temp.state.json"
    
    if os.path.exists(temp_state_file):
        session_files.append(temp_state_file)
    
    if not session_files:
        print("No session state files found.")
        return
    
    # Check sizes
    large_files = []
    for file_path in session_files:
        if os.path.exists(file_path):
            size_mb = os.path.getsize(file_path) / (1024 * 1024)  # Size in MB
            print(f"{file_path}: {size_mb:.2f} MB")
            
            if size_mb > 100:  # Consider files over 100MB as large
                large_files.append((file_path, size_mb))
    
    # Report any large files
    if large_files:
        print(f"\nFound {len(large_files)} large session state files:")
        for file_path, size_mb in large_files:
            print(f"- {file_path}: {size_mb:.2f} MB")
        
        if input("\nDo you want to clean these files? (y/n): ").lower() == 'y':
            run_cleanup_script()
    else:
        print("\nNo large session state files found.")

def run_cleanup_script():
    # Check if the cleanup script exists
    cleanup_script = "./clean_session_state.py"
    if not os.path.exists(cleanup_script):
        print(f"Error: Cleanup script not found at {cleanup_script}")
        return
    
    # Run the cleanup script
    try:
        print("\nRunning cleanup script...")
        subprocess.run(["python", cleanup_script], check=True)
        print("\nCleanup completed successfully.")
        print("You should restart the Streamlit app to use the cleaned state.")
    except subprocess.CalledProcessError as e:
        print(f"Error running cleanup script: {e}")
        
if __name__ == "__main__":
    main() 