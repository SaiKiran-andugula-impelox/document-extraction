import streamlit as st
from pdf2image import convert_from_bytes
from PIL import Image
import io
import logging

from get_descriptions_and_confidence import get_descriptions_and_confidence


# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def extract_images_from_pdf(uploaded_file):
    """
    Processes the uploaded PDF file by converting each page to an image, extracting descriptions, 
    and calculating confidence scores.

    Args:
        uploaded_file (file-like object): The uploaded PDF file to be processed.

    Returns:
        tuple: A tuple containing (descriptions, confidence_result), where descriptions are the extracted 
               text/image descriptions, and confidence_result is a dictionary with confidence level.
               Returns None if any error occurs.

    Raises:
        None: Any exception is logged and handled gracefully without propagating.
    
    Process:
        1. Converts the uploaded PDF to a list of images.
        2. Extracts descriptions and confidence levels from the images.
        3. Displays relevant messages and logs errors when necessary.
    """
    try:
        logging.info("Starting PDF processing.")

        # Step 1: Convert PDF pages to images
        pdf_images = convert_from_bytes(uploaded_file.getbuffer(), fmt='jpeg')
        image_bytes_list = []

        for page_num, img in enumerate(pdf_images, start=1):
            try:
                logging.info(f"Processing page {page_num} of PDF.")
                
                # Convert each image into a byte array
                img_byte_arr = io.BytesIO()
                img.save(img_byte_arr, format='JPEG')  # Save as JPEG in memory
                image_bytes_list.append(img_byte_arr.getvalue())
            
            except Exception as e:
                logging.error(f"Image encoding failed for page {page_num}: {str(e)}")
                continue  # Skip the page if image encoding fails

        if not image_bytes_list:
            logging.error("No images could be processed from the PDF.")
            return None
        
        # Step 2: Extract descriptions and confidence from images
        descriptions, confidence_result = get_descriptions_and_confidence(image_bytes_list, is_image_type=True)
        
        logging.info("Successfully processed PDF and extracted descriptions and confidence.")
        return descriptions, confidence_result

    except Exception as e:
        logging.error(f"Error during PDF processing: {str(e)}")
        return None
