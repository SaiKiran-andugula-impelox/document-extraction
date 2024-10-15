import fitz
from pdf2image import convert_from_bytes
import logging

# Set up logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def check_for_images_in_pdf(uploaded_file):
    """
    Checks if a PDF contains images and whether those images are in acceptable image modes (RGB/RGBA).

    Args:
        uploaded_file (file-like object): The uploaded PDF file to be processed.

    Process Details:
        - uploaded_file.read(): Creates a binary data of the uploaded file. Even a small image (e.g., stamp/logo) is considered an image.
        - uploaded_file.seek(0): After reading the file, the file pointer is at the end, so it needs to be reset to the beginning.
        - uploaded_file.buffer(): Creates a memory view of the file without copying its contents.

    Returns:
        bool: Returns True if images are found in acceptable image modes (RGB/RGBA).
            # In some cases, skipping this step (Step 2) could cause the system to misinterpret the content of the PDF. Even if the PDF contains images, 
            # the model might incorrectly consider it as text,and it won't be able to extract the image content properly. 
            # Therefore, ensure that all PDFs with images go through this image conversion step for accurate processing.

    Raises:
        Exception: Logs any exceptions encountered during PDF image extraction or conversion.
    """
    try:
        # Step 1: Check if the PDF contains images using PyMuPDF (fitz)
        pdf_document = fitz.open(stream=uploaded_file.read(), filetype="pdf")
        has_images = False

        # Check each page for images
        for page in pdf_document:
            images = page.get_images(full=True)
            if images:
                has_images = True
                logging.info("Images found in the PDF.")
                break
        
        uploaded_file.seek(0)

        if not has_images:
            logging.info("No images found in the PDF.")
            return False  
        
        # Step 2: If images are found, convert the PDF to images and check image modes
        try:
            pdf_images = convert_from_bytes(uploaded_file.getbuffer())
        except Exception as e:
            logging.error(f"Error converting PDF to images: {str(e)}")
            return False  # Handle PDF conversion errors
        
        for image in pdf_images:
            if image.mode in ["RGB", "RGBA"]:  # Check for acceptable image modes, 
                logging.info("Images found in acceptable modes (RGB/RGBA).")
                return True  
        
        logging.info("No images with acceptable modes found.")
        return False  
    
    except Exception as e:
        logging.error(f"An error occurred while checking images in the PDF: {str(e)}")
        return False  # Handle any unexpected errors
