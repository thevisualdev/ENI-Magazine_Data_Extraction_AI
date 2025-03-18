#!/usr/bin/env python3
import file_manager
import yaml

def test_file_path(file_path):
    """Test metadata extraction for a specific file path."""
    print(f"\nTesting extraction for: {file_path}")
    
    # Load config
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Extract metadata
    metadata = file_manager.extract_metadata_from_path(file_path, config)
    
    # Print results
    print(f"\nFinal extracted metadata:")
    print(f"  Magazine: '{metadata['magazine']}'")
    print(f"  Issue: '{metadata['magazine_no']}'")
    print(f"  Author: '{metadata['author']}'")
    print(f"  Title: '{metadata['title']}'")
    print("\n" + "-"*50)

if __name__ == "__main__":
    # Test with a list of diverse file paths to check different patterns
    test_files = [
        # Original problematic file
        'data/Orizzonti/Orizzonti_62_x_web/Emergenza Idrica/12_13_Emergenza Idrica .docx',
        
        # Files with different patterns
        'data/WE/WE 57/Pistelli/12_17_Pistelli.docx',                                 # Space in issue folder
        'data/Orizzonti/Orizzonti_61_x_web/Intro_Lucia/02_03_Intro_Lucia.docx',       # Underscore in author
        'data/WE/We 48/Graziani/Graziani.docx',                                       # Lowercase "we"
        'data/WE/We 49/Lifan li/Lifan_Li.docx',                                       # Space in author name
        'data/WE/WE_62/Arcesati/62_69_Arcesati.docx',                                 # Standard pattern
        'data/WE/WE_60/92_95_Franceschini/92_95_Franceschini.docx'                    # Author folder includes numbers
    ]
    
    for file_path in test_files:
        test_file_path(file_path)
