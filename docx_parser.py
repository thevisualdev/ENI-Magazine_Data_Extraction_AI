import os
import io
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
        - preview_image_path: Path to the saved preview image (if any), or None
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
        preview_image_path = None
        
        # Create a directory for images next to the document
        doc_dir = os.path.dirname(file_path)
        images_dir = os.path.join(doc_dir, "__images__")
        os.makedirs(images_dir, exist_ok=True)
        
        # Get document name without extension for the image filename
        doc_name = os.path.splitext(os.path.basename(file_path))[0]
        
        # Check for images in the document
        for rel in doc.part.rels.values():
            if "image" in rel.target_ref:
                try:
                    # Get image data
                    image_data = rel.target_part.blob
                    
                    # Get image type from rel.target_ref (e.g., "image/jpeg")
                    image_type = rel.target_ref.split('/')[-1]  # Extract extension
                    
                    # Create a filename for the image
                    image_filename = f"{doc_name}_preview.{image_type}"
                    image_path = os.path.join(images_dir, image_filename)
                    
                    # Save the image to disk
                    with open(image_path, 'wb') as img_file:
                        img_file.write(image_data)
                    
                    preview_image_path = image_path
                    
                    # Only use the first image as preview
                    break
                except Exception as e:
                    print(f"Error extracting image: {e}")
        
        return text_content, preview_image_path
    
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