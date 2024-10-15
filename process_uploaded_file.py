import logging
import streamlit as st

from check_for_images_in_pdf import check_for_images_in_pdf
from extract_images_from_pdf import extract_images_from_pdf
from process_image import process_image
from extract_text_from_pdf import extract_text_from_pdf


def process_uploaded_file(uploaded_file):
    """
    Processes an uploaded file by determining its type (PDF or image) and 
    extracting content accordingly.

    Args:
        uploaded_file (file-like object): The uploaded file to be processed.

    Returns:
        None: The function processes the file in place and logs the output.

    Raises:
        Exception: Logs any exceptions encountered during file processing.
    """
    try:
        if uploaded_file is not None:
            logging.info("File uploaded successfully. Checking file type.")

            if uploaded_file.type == "application/pdf":
                logging.info("Processing PDF file.")
                if check_for_images_in_pdf(uploaded_file):
                    logging.info("Images found in PDF. Extracting images.")
                    extract_images_from_pdf(uploaded_file)
                else:
                    logging.info("No images found in PDF. Extracting text.")
                    extract_text_from_pdf(uploaded_file)
            else:
                logging.info("Processing image file.")
                process_image(uploaded_file)
        else:
            logging.warning("No file uploaded.")
    except Exception as e:
        logging.error(f"Error processing uploaded file: {str(e)}")


# Streamlit file uploader
uploaded_file = st.file_uploader("Choose an image or PDF...", type=["jpg", "jpeg", "png", "pdf"])
process_uploaded_file(uploaded_file)