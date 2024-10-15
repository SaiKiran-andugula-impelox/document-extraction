import cv2
import numpy as np
import streamlit as st
import openai
import json
from pdf2image import convert_from_bytes
import os  # For accessing environment variables



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





