import os
import pandas as pd
from typing import List, Dict, Any, Optional

def create_dataframe(file_data_list: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Create a DataFrame from a list of file data dictionaries.
    
    Args:
        file_data_list: List of file data dictionaries
        
    Returns:
        DataFrame containing all metadata fields
    """
    # First determine all possible columns across all dictionaries
    all_columns = set()
    for file_data in file_data_list:
        all_columns.update(file_data.keys())
    
    # Remove fields we don't want to include in the CSV
    exclude_fields = ['text_content', 'preview_image']
    for field in exclude_fields:
        if field in all_columns:
            all_columns.remove(field)
    
    # Create a list of dictionaries with only the selected columns
    rows = []
    for file_data in file_data_list:
        row = {col: file_data.get(col, '') for col in all_columns}
        
        # Handle preview_image_path
        if 'preview_image_path' not in row:
            # Set preview_image_path based on available images
            if 'folder_images' in file_data and file_data['folder_images']:
                if isinstance(file_data['folder_images'], list) and file_data['folder_images']:
                    row['preview_image_path'] = file_data['folder_images'][0]
                elif isinstance(file_data['folder_images'], str) and file_data['folder_images']:
                    paths = file_data['folder_images'].split('|')
                    if paths:
                        row['preview_image_path'] = paths[0]
                else:
                    row['preview_image_path'] = ''
            elif 'preview_image' in file_data and file_data['preview_image'] and not isinstance(file_data['preview_image'], str):
                # Preview is something other than base64
                row['preview_image_path'] = str(file_data['preview_image'])
            else:
                row['preview_image_path'] = ''
        
        # Always include full_path for reference even if it's already in all_columns
        row['full_path'] = file_data.get('full_path', '')
        
        rows.append(row)
    
    # Create DataFrame with specific data types to avoid conversion issues
    df = pd.DataFrame(rows)
    
    # Convert all columns to string type to avoid data type issues
    for col in df.columns:
        df[col] = df[col].astype(str)
    
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
    mask = df['full_path'].astype(str) == str(file_data.get('full_path', ''))
    
    # If full_path is missing, fall back to the old method
    if not file_data.get('full_path', '') or not mask.any():
        mask = (
            (df['title'].astype(str) == str(file_data.get('title', ''))) & 
            (df['magazine'].astype(str) == str(file_data.get('magazine', ''))) & 
            (df['author'].astype(str) == str(file_data.get('author', '')))
        )
    
    # Convert folder_images list to string if present
    folder_images_str = ""
    if 'folder_images' in file_data and file_data['folder_images']:
        # If it's a list, limit to max 10 images to prevent excessive data
        if isinstance(file_data['folder_images'], list):
            # Only store paths, not full image data
            image_list = file_data['folder_images'][:10] if len(file_data['folder_images']) > 10 else file_data['folder_images']
            folder_images_str = "|".join(image_list)
        # If it's already a string, keep it as is
        elif isinstance(file_data['folder_images'], str):
            folder_images_str = file_data['folder_images']
    
    # Handle preview image - store path instead of base64 data
    preview_image_path = ""
    if 'folder_images' in file_data and file_data['folder_images']:
        # If we have folder images, use the first one as preview path
        if isinstance(file_data['folder_images'], list) and file_data['folder_images']:
            preview_image_path = file_data['folder_images'][0]
        elif isinstance(file_data['folder_images'], str) and file_data['folder_images']:
            paths = file_data['folder_images'].split('|')
            if paths:
                preview_image_path = paths[0]
    elif 'preview_image' in file_data and file_data['preview_image'] and not file_data['preview_image'].startswith('data:'):
        # If preview_image is already a path, use it
        preview_image_path = file_data['preview_image']
    else:
        # Mark that there was an embedded image but we're not storing it
        preview_image_path = "[embedded image in document]"
    
    # Create a new row with the file data
    row = {
        'title': str(file_data.get('title', '')),
        'abstract': str(file_data.get('abstract', '')),
        'abstract_ita': str(file_data.get('abstract_ita', '')),
        'abstract_eng': str(file_data.get('abstract_eng', '')),
        'language': str(file_data.get('language', '')),
        'preview_image_path': preview_image_path,  # Store path instead of base64
        'folder_images': folder_images_str,
        'magazine': str(file_data.get('magazine', '')),
        'magazine_no': str(file_data.get('magazine_no', '')),
        'theme': str(file_data.get('theme', '')),
        'format': str(file_data.get('format', '')),
        'author': str(file_data.get('author', '')),
        'geographic_area': str(file_data.get('geographic_area', '')),
        'keywords': str(file_data.get('keywords', '')),
        'original_magazine': str(file_data.get('original_magazine', file_data.get('magazine', ''))),
        'original_magazine_no': str(file_data.get('original_magazine_no', file_data.get('magazine_no', ''))),
        'original_author': str(file_data.get('original_author', file_data.get('author', ''))),
        'original_title': str(file_data.get('original_title', file_data.get('title', ''))),
        'full_path': str(file_data.get('full_path', ''))
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
    
    # Ensure all columns are strings to avoid type issues
    for col in df.columns:
        df[col] = df[col].astype(str)
    
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
            # Try to read the CSV file with string dtypes to prevent automatic type conversion
            df = pd.read_csv(file_path, encoding='utf-8', dtype=str)
            
            # Make sure all columns are explicitly cast to string to avoid typing issues
            for col in df.columns:
                df[col] = df[col].astype(str)
                
            # Handle column name changes for backward compatibility
            if 'preview_image' in df.columns and 'preview_image_path' not in df.columns:
                # Rename or add preview_image_path for compatibility
                if df['preview_image'].str.startswith('data:').any():
                    # Some entries contain base64 data, mark them as embedded
                    df['preview_image_path'] = df['preview_image'].apply(
                        lambda x: "[embedded image in document]" if str(x).startswith('data:') else x
                    )
                else:
                    # All entries are likely paths already, just copy them
                    df['preview_image_path'] = df['preview_image']
                
                # Remove the preview_image column to save space
                df = df.drop(columns=['preview_image'])
                
            return df
        except pd.errors.EmptyDataError:
            # Handle case where the file exists but is empty or has no columns
            print(f"Warning: CSV file at {file_path} is empty or has no valid columns.")
            # Return an empty DataFrame with the expected columns
            return pd.DataFrame(columns=[
                'title', 'abstract', 'abstract_ita', 'abstract_eng', 'language', 'preview_image_path', 
                'magazine', 'magazine_no', 'theme', 'format', 'author', 'geographic_area', 'keywords',
                'original_magazine', 'original_magazine_no', 'original_author', 'original_title',
                'full_path', 'folder_images'
            ])
        except Exception as e:
            # Handle other exceptions when reading the file
            print(f"Error reading CSV file at {file_path}: {e}")
            # Return None to indicate an error
            return None
    else:
        # File doesn't exist
        return None 