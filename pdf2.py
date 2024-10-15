import cv2
import numpy as np
import streamlit as st
import openai
import json
import asyncio
from pdf2image import convert_from_bytes
import os  # For accessing environment variables



from get_image_description import get_image_description
from get_confidence_level import get_confidence_level
from extract_text_from_pdf import extract_text_from_pdf
from check_for_images_in_pdf import check_for_images_in_pdf
from extract_images_from_pdf import extract_images_from_pdf


# Load the environment variables from the .env file
from dotenv import load_dotenv  # For loading the .env file
load_dotenv()


# Access the OpenAI API key from the environment variables
openai.api_key = os.getenv("OPEN_API_KEY")


# Updated prompt for processing and analyzing textual data
system_prompt = (
    """
    "You are a highly skilled and detail-oriented assistant specialized in processing and analyzing textual data extracted from PDFs. "
    "Your primary task is to help the user extract specific details, numerical values, or other relevant information from the text content provided. "
    "The user may ask questions or make requests related to various aspects of the extracted text, including counting items, identifying categories, extracting personal details, and more.\n\n"
    "When responding to the user's requests:\n"
    "1. Understand the Context: Carefully read the user's request to ensure you fully understand what information they need and do not extract the information which is not in the prompt\n"
    "2. Text Extraction and Analysis: Analyze the provided text to extract accurate information, such as numerical values, names, categories, or other details.\n"
    "3. Clear and Concise Responses: Provide clear, concise, and accurate responses based on the text content. Include relevant details and context to ensure the user gets the exact information they need.\n"
    "4. Highlight Key Information: When listing items or details, organize them in a structured format (e.g., bullet points or numbered lists) for easy readability.\n"
    "5. Accuracy and Verification: Double-check your analysis to ensure the accuracy of the extracted information, especially when dealing with numerical data or critical details.\n"
    "6. Handle Complex Queries: If the user's query is complex or involves multiple steps, break down the response into logical parts and guide the user through each step.\n"
    "7. Your answers should strictly like an object with key-value pairs. There can be other data structure contain inside the object as well if necessary. Add descriptions if only user asks for it. If user asks for certain values, provide object like structure.\n\n"
    "8. Extract only the particular key-value pair the user is asking for."
    """
)

# Streamlit app layout
st.title("Document extraction..")

st.write("Upload an image or PDF and get a description using GPT-4o.")

# Textbox for updating the prompt
user_prompt = st.text_input("Enter the prompt for image description", "What’s in the document?")

# Upload image or PDF button



# Main processing logic
def  process_uploaded_file(uploaded_file):

    if uploaded_file is not None:
        if uploaded_file.type == "application/pdf":
            st.write("Processing PDF...")

            # Check if the PDF contains images in acceptable modes
            if check_for_images_in_pdf(uploaded_file):
                st.write("Images found in the PDF. Processing images...")

                # Convert PDF pages with images to image format and get descriptions
                image_bytes_array = process_pdf_images(uploaded_file)

                list_all_descriptions = []
                for _ in range(3):
                    all_descriptions = get_image_description(openai, image_bytes_array, system_prompt=system_prompt, user_prompt=user_prompt, is_image_type=True)
                    list_all_descriptions.append(all_descriptions)     

                # Display all image descriptions
                st.write("Image descriptions from the PDF:")
                st.write(list_all_descriptions)

                # Get confidence level for the descriptions
                list_all_descriptions_str = json.dumps(list_all_descriptions)
                confidence_result = get_confidence_level(openai, list_all_descriptions_str)
                st.write(confidence_result)

            else:
                st.write("No images found in the PDF or no valid images. Extracting text...")

                # Extract text from PDF and collect all pages into a list
                pdf_text_descriptions = pdf_text(uploaded_file)
                
                if pdf_text_descriptions:  # Check if there are any extracted texts
                    st.write("Extracting text from the PDF:")

                    # Prepare to store descriptions from the LLM
                    list_pdf_text_descriptions = []

                    # Loop to get image descriptions based on extracted text
                    for _ in range(3):
                        all_descriptions = get_image_description(openai, pdf_text_descriptions, system_prompt=system_prompt, user_prompt=user_prompt, is_image_type=False)
                        list_pdf_text_descriptions.append(all_descriptions)
                    
                    st.write("text descriptions from the PDF:")
                    st.write(list_pdf_text_descriptions)
                    
                    # Convert the list of descriptions to JSON for confidence analysis
                    list_pdf_text_descriptions_str = json.dumps(list_pdf_text_descriptions) 
                    confidence_result = get_confidence_level(openai, list_pdf_text_descriptions_str)
                    st.write(confidence_result)
                else:
                    st.write("No text found in the PDF.")

        else:
            # Handle image file (JPEG/PNG)
            img_before = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR)

            if img_before is None: # Check if image decoding was successful from corupted image, unsupported format
                st.error("Error loading image. Please upload a valid image file.")
            else:
                img_gray = cv2.cvtColor(img_before, cv2.COLOR_BGR2GRAY)

                if np.var(img_gray) < 1000: 
                    st.error("The image is not readable. Please upload a clearer image.")
                else:
                    st.write("Classifying...")
                    descriptions = []

                    for _ in range(3):
                        description = get_image_description(openai, uploaded_file.getbuffer(),  system_prompt=system_prompt, user_prompt=user_prompt, is_image_type=True)
                        descriptions.append(description)

                    #st.write(descriptions)

                    descriptions_str = json.dumps(descriptions)
                    result = get_confidence_level(openai, descriptions_str)
                    st.write(result)

uploaded_file = st.file_uploader("Choose an image or PDF...", type=["jpg", "jpeg", "png", "pdf"])
print(process_uploaded_file(uploaded_file))