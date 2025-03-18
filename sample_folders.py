#!/usr/bin/env python3
import os
import random
import argparse
from typing import List

def find_all_docx_files(root_dir: str) -> List[str]:
    """Find all .docx files in the directory tree."""
    docx_files = []
    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.lower().endswith('.docx'):
                full_path = os.path.join(root, file)
                docx_files.append(full_path)
    return docx_files

def display_folder_structure(file_paths: List[str]):
    """Display the folder structure of the given file paths."""
    for i, file_path in enumerate(file_paths, 1):
        # Get relative path from current directory
        try:
            rel_path = os.path.relpath(file_path)
        except ValueError:
            # Handle case where paths might be on different drives
            rel_path = file_path
        
        # Split into components
        components = rel_path.split(os.sep)
        
        print(f"\n{i}. File: {os.path.basename(file_path)}")
        print(f"   Full path: {file_path}")
        print(f"   Components:")
        
        # Print each component of the path
        for j, component in enumerate(components):
            if j == len(components) - 1:  # This is the filename
                print(f"     {j}. [FILE] {component}")
            else:
                print(f"     {j}. [DIR] {component}")
        
        # Extract potential magazine and issue number if they exist
        if len(components) >= 3:
            potential_magazine_dir = components[1] if components[0] == "data" else components[0]
            print(f"   Potential Magazine: {potential_magazine_dir}")
            
            if len(components) >= 4:
                potential_issue_dir = components[2] if components[0] == "data" else components[1]
                print(f"   Potential Issue: {potential_issue_dir}")

def main():
    parser = argparse.ArgumentParser(description="Sample random .docx files and display their folder structures.")
    parser.add_argument("root_dir", default="data", nargs="?", help="Root directory to search (default: data)")
    parser.add_argument("-n", "--num_samples", type=int, default=5, help="Number of random files to sample (default: 5)")
    parser.add_argument("--seed", type=int, help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    # Set random seed if provided
    if args.seed is not None:
        random.seed(args.seed)
    
    # Find all docx files
    print(f"Searching for .docx files in {args.root_dir}...")
    all_docx_files = find_all_docx_files(args.root_dir)
    total_files = len(all_docx_files)
    
    if total_files == 0:
        print(f"No .docx files found in {args.root_dir}")
        return
    
    print(f"Found {total_files} .docx files")
    
    # Sample random files
    num_samples = min(args.num_samples, total_files)
    sampled_files = random.sample(all_docx_files, num_samples)
    
    print(f"\nDisplaying folder structure for {num_samples} randomly sampled files:")
    display_folder_structure(sampled_files)
    
    print("\nAnalysis Complete.")

if __name__ == "__main__":
    main()
