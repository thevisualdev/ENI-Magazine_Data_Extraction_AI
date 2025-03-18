import os
import pandas as pd
from typing import List, Dict, Any, Optional

def create_dataframe(file_data_list: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Create a DataFrame from the list of file data dictionaries.
    
    Args:
        file_data_list: List of dictionaries containing file metadata and extracted fields
        
    Returns:
        Pandas DataFrame with all the data
    """
    # Define the columns for the DataFrame
    columns = [
        'title', 'abstract', 'preview_image', 'magazine', 'magazine_no',
        'theme', 'format', 'author', 'geographic_area', 'keywords',
        'original_magazine', 'original_magazine_no', 'original_author', 'original_title',
        'full_path'
    ]
    
    # Create an empty DataFrame with the defined columns
    df = pd.DataFrame(columns=columns)
    
    # Add data from each file to the DataFrame
    for file_data in file_data_list:
        # Create a new row with only the columns we want
        row = {
            'title': file_data.get('title', ''),
            'abstract': file_data.get('abstract', ''),
            'preview_image': file_data.get('preview_image', ''),
            'magazine': file_data.get('magazine', ''),
            'magazine_no': file_data.get('magazine_no', ''),
            'theme': file_data.get('theme', ''),
            'format': file_data.get('format', ''),
            'author': file_data.get('author', ''),
            'geographic_area': file_data.get('geographic_area', ''),
            'keywords': file_data.get('keywords', ''),
            'original_magazine': file_data.get('original_magazine', file_data.get('magazine', '')),
            'original_magazine_no': file_data.get('original_magazine_no', file_data.get('magazine_no', '')),
            'original_author': file_data.get('original_author', file_data.get('author', '')),
            'original_title': file_data.get('original_title', file_data.get('title', '')),
            'full_path': file_data.get('full_path', '')
        }
        
        # Append the row to the DataFrame
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    
    return df

def save_to_csv(df: pd.DataFrame, output_path: str) -> str:
    """
    Save the DataFrame to a CSV file.
    
    Args:
        df: The DataFrame to save
        output_path: The path where to save the CSV file
        
    Returns:
        The path to the saved CSV file
    """
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save the DataFrame to CSV
    df.to_csv(output_path, index=False, encoding='utf-8')
    
    return output_path

def update_dataframe(df: pd.DataFrame, file_data: Dict[str, Any]) -> pd.DataFrame:
    """
    Update a DataFrame with new file data.
    
    Args:
        df: The DataFrame to update
        file_data: The file data dictionary to add or update
        
    Returns:
        The updated DataFrame
    """
    # Check if the file already exists in the DataFrame
    # We identify files by the full path to avoid conflicts with renamed or corrected metadata
    mask = df['full_path'] == file_data.get('full_path', '')
    
    # If full_path is missing, fall back to the old method
    if not file_data.get('full_path', '') or not mask.any():
        mask = (
            (df['title'] == file_data.get('title', '')) & 
            (df['magazine'] == file_data.get('magazine', '')) & 
            (df['author'] == file_data.get('author', ''))
        )
    
    # Create a new row with the file data
    row = {
        'title': file_data.get('title', ''),
        'abstract': file_data.get('abstract', ''),
        'preview_image': file_data.get('preview_image', ''),
        'magazine': file_data.get('magazine', ''),
        'magazine_no': file_data.get('magazine_no', ''),
        'theme': file_data.get('theme', ''),
        'format': file_data.get('format', ''),
        'author': file_data.get('author', ''),
        'geographic_area': file_data.get('geographic_area', ''),
        'keywords': file_data.get('keywords', ''),
        'original_magazine': file_data.get('original_magazine', file_data.get('magazine', '')),
        'original_magazine_no': file_data.get('original_magazine_no', file_data.get('magazine_no', '')),
        'original_author': file_data.get('original_author', file_data.get('author', '')),
        'original_title': file_data.get('original_title', file_data.get('title', '')),
        'full_path': file_data.get('full_path', '')
    }
    
    # If the file already exists, update it, otherwise append it
    if mask.any():
        # Get the index of the row to update
        index = df.index[mask].tolist()[0]
        
        # Update the row
        for key, value in row.items():
            # Only update if the column exists in the DataFrame
            if key in df.columns:
                df.at[index, key] = value
            else:
                # Add new column if it doesn't exist
                df[key] = None
                df.at[index, key] = value
    else:
        # Append a new row
        new_df = pd.DataFrame([row])
        
        # Ensure all columns from df are in new_df
        for col in df.columns:
            if col not in new_df.columns:
                new_df[col] = None
        
        # Append the new row
        df = pd.concat([df, new_df], ignore_index=True)
    
    return df

def read_csv(file_path: str) -> Optional[pd.DataFrame]:
    """
    Read a CSV file into a DataFrame.
    
    Args:
        file_path: The path to the CSV file
        
    Returns:
        The DataFrame or None if the file doesn't exist or is empty
    """
    if os.path.exists(file_path):
        try:
            # Try to read the CSV file
            df = pd.read_csv(file_path, encoding='utf-8')
            return df
        except pd.errors.EmptyDataError:
            # Handle case where the file exists but is empty or has no columns
            print(f"Warning: CSV file at {file_path} is empty or has no valid columns.")
            # Return an empty DataFrame with the expected columns
            return pd.DataFrame(columns=[
                'title', 'abstract', 'preview_image', 'magazine', 'magazine_no',
                'theme', 'format', 'author', 'geographic_area', 'keywords',
                'original_magazine', 'original_magazine_no', 'original_author', 'original_title',
                'full_path'
            ])
        except Exception as e:
            # Handle other exceptions when reading the file
            print(f"Error reading CSV file at {file_path}: {e}")
            # Return None to indicate an error
            return None
    else:
        # File doesn't exist
        return None 