import os
import io
import base64
from typing import Dict, Any, Tuple, Optional, List
from docx import Document
from PIL import Image

def extract_docx_content(file_path: str) -> Tuple[str, Optional[str]]:
    """
    Extract text content and preview image from a DOCX file.
    
    Args:
        file_path: Path to the DOCX file
        
    Returns:
        Tuple containing:
        - text: The extracted text content
        - preview_image: Base64 encoded string of the first image (if any), or None
    """
    try:
        doc = Document(file_path)
        
        # Extract text content
        paragraphs = []
        for para in doc.paragraphs:
            if para.text.strip():
                paragraphs.append(para.text)
        
        text_content = "\n".join(paragraphs)
        
        # Extract first image (if any)
        preview_image = None
        
        # Check for images in the document
        for rel in doc.part.rels.values():
            if "image" in rel.target_ref:
                try:
                    # Get image data
                    image_data = rel.target_part.blob
                    
                    # Convert to base64 for storage/display
                    image_base64 = base64.b64encode(image_data).decode('utf-8')
                    
                    # Get image type from rel.target_ref (e.g., "image/jpeg")
                    image_type = rel.target_ref.split('/')[-1]  # Extract extension
                    
                    # Create base64 string with image type
                    preview_image = f"data:image/{image_type};base64,{image_base64}"
                    
                    # Only use the first image as preview
                    break
                except Exception as e:
                    print(f"Error extracting image: {e}")
        
        return text_content, preview_image
    
    except Exception as e:
        print(f"Error parsing DOCX file {file_path}: {e}")
        return "", None
        
def save_preview_image(image_base64: str, output_dir: str, file_name: str) -> str:
    """
    Save a base64 encoded image to the output directory.
    
    Args:
        image_base64: Base64 encoded image string
        output_dir: Directory to save the image
        file_name: Base name for the image file
        
    Returns:
        Path to the saved image
    """
    if not image_base64:
        return ""
    
    try:
        # Make sure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Extract the image data (remove the data:image/xxx;base64, prefix)
        image_data = image_base64.split(',')[1]
        
        # Decode base64 data
        image_bytes = base64.b64decode(image_data)
        
        # Open image using PIL
        image = Image.open(io.BytesIO(image_bytes))
        
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
    
    Args:
        file_path: Path to the DOCX file
        
    Returns:
        List of full paths to image files in the folder
    """
    folder_path = os.path.dirname(file_path)
    image_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp')
    images = []
    
    try:
        for file in os.listdir(folder_path):
            if file.lower().endswith(image_extensions):
                image_path = os.path.join(folder_path, file)
                images.append(image_path)
                
        # Sort images by name to ensure consistent order
        images.sort()
        
        return images
    except Exception as e:
        print(f"Error scanning folder for images: {e}")
        return [] 