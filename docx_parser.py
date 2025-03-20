import os
import io
from typing import Dict, Any, Tuple, Optional, List
from docx import Document
from PIL import Image

def extract_docx_content(file_path: str) -> Tuple[str, Optional[str]]:
    """
    Extract text content from a DOCX file.
    
    Args:
        file_path: Path to the DOCX file
        
    Returns:
        Tuple containing:
        - text: The extracted text content
        - preview_image_path: Always None since we don't extract images
    """
    # Skip .AppleDouble files and metadata files
    if '.AppleDouble' in file_path or '/.' in file_path:
        print(f"Skipping system file: {file_path}")
        return "", None
        
    try:
        doc = Document(file_path)
        
        # Extract text content
        paragraphs = []
        for para in doc.paragraphs:
            if para.text.strip():
                paragraphs.append(para.text)
        
        text_content = "\n".join(paragraphs)
        
        # We don't need to extract images from the document anymore
        return text_content, None
    
    except Exception as e:
        print(f"Error parsing DOCX file {file_path}: {e}")
        return "", None

def save_preview_image(image_data: bytes, output_dir: str, file_name: str) -> str:
    """
    Save image data to the output directory.
    
    Args:
        image_data: Binary image data
        output_dir: Directory to save the image
        file_name: Base name for the image file
        
    Returns:
        Path to the saved image
    """
    if not image_data:
        return ""
    
    try:
        # Make sure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Open image using PIL
        image = Image.open(io.BytesIO(image_data))
        
        # Save the image
        output_path = os.path.join(output_dir, f"{file_name}.png")
        image.save(output_path)
        
        return output_path
    
    except Exception as e:
        print(f"Error saving preview image: {e}")
        return ""

def find_folder_images(file_path: str) -> List[str]:
    """
    Find all image files in the same folder as the DOCX file.
    Returns relative paths rather than absolute paths.
    
    Args:
        file_path: Path to the DOCX file
        
    Returns:
        List of relative paths to image files in the folder
    """
    folder_path = os.path.dirname(file_path)
    image_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp')
    images = []
    
    # Skip .AppleDouble directories and look at the parent directory instead
    if '.AppleDouble' in folder_path:
        # Move up one directory to get the actual content folder
        folder_path = os.path.dirname(folder_path)
    
    try:
        # Check if directory exists before listing
        if not os.path.exists(folder_path):
            print(f"Warning: Folder doesn't exist: {folder_path}")
            return []
            
        # Get the project base directory for making relative paths
        # Assuming a typical structure with /data/ as a subdirectory
        base_dir = folder_path
        if '/data/' in folder_path:
            base_dir = folder_path.split('/data/')[0]
        
        # First look in the document's directory
        for file in os.listdir(folder_path):
            if file.lower().endswith(image_extensions):
                image_path = os.path.join(folder_path, file)
                # Verify the file exists and is accessible
                if os.path.isfile(image_path) and os.access(image_path, os.R_OK):
                    # Store path relative to project root if possible
                    if image_path.startswith(base_dir):
                        rel_path = os.path.relpath(image_path, base_dir)
                        images.append(rel_path)
                    else:
                        images.append(image_path)
        
        # Sort images by name to ensure consistent order
        images.sort()
        
        # Log the result for debugging
        if images:
            print(f"Found {len(images)} image(s) for document: {file_path}")
            for img in images[:3]:  # Print first 3 for brevity
                print(f"  - {img}")
            if len(images) > 3:
                print(f"  - ... and {len(images) - 3} more")
        else:
            print(f"No images found for document: {file_path}")
            
        return images
    except Exception as e:
        print(f"Error scanning folder for images: {e}")
        return [] 