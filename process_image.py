import cv2
import numpy as np
import logging
import streamlit as st
from get_descriptions_and_confidence import get_descriptions_and_confidence

def process_image(uploaded_file):
    """
    Processes an uploaded image file by decoding it and extracting descriptions
    and confidence levels. It checks for image validity and clarity before processing.

    Args:
        uploaded_file (file-like object): The uploaded image file to be processed.

    Returns:
        None: The function processes the image in place and logs the output.

    Raises:
        Exception: Logs any exceptions encountered during image processing.
    """
    try:
        logging.info("Attempting to decode the uploaded image.")
        img_before = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR)

        if img_before is None:
            st.error("Error loading image. Please upload a valid image file.")
            logging.error("Image decoding failed; uploaded file may be corrupted or unsupported format.")
            return
        
        img_gray = cv2.cvtColor(img_before, cv2.COLOR_BGR2GRAY)

        if np.var(img_gray) < 1000:
            st.error("The image is not readable. Please upload a clearer image.")
            logging.warning("Uploaded image is unclear; variance in grayscale is too low.")
            return
        
        st.write("Classifying...")
        descriptions, confidence_result = get_descriptions_and_confidence(uploaded_file.getbuffer(), is_image_type=True)

        # Displaying the confidence level result
        # st.write("Confidence level result")
        # st.write(confidence_result)

    except Exception as e:
        logging.error(f"Error processing image: {str(e)}")
