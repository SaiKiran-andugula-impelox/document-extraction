import fitz
import logging

from get_descriptions_and_confidence import get_descriptions_and_confidence

def extract_text_from_pdf(uploaded_file):
    """
    Extracts text from a PDF file and processes it to obtain descriptions and confidence levels.

    Args:
        uploaded_file (file-like object): The uploaded PDF file to extract text from.

    Returns:
        tuple: A tuple containing (descriptions, confidence_result). If text extraction fails, returns None.

    Raises:
        None: Any exceptions are logged internally.
    """
    try:
        logging.info("Starting PDF text extraction.")
        
        # Extract text from the PDF
        with fitz.open(stream=uploaded_file.read(), filetype="pdf") as doc:
            logging.info(f"Processing {doc.page_count} pages.")
            text_descriptions = ', '.join(page.get_text() for page in doc)
        
        if not text_descriptions:
            logging.warning("No text found in the PDF.")
            return None
        
        logging.info("Text successfully extracted. Calculating descriptions and confidence levels.")
        descriptions, confidence_result = get_descriptions_and_confidence(text_descriptions, is_image_type=False)
        
        logging.info("Descriptions and confidence levels calculated successfully.")
        return descriptions, confidence_result
    
    except Exception as e:
        logging.error(f"Error during PDF text processing: {str(e)}")
        return None
