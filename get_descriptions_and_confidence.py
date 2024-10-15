
from get_image_description import get_image_description
from get_confidence_level import get_confidence_level
import openai
import json
import streamlit as st


from main import system_prompt, user_prompt



def get_descriptions_and_confidence(data, is_image_type):
    descriptions = []
    for _ in range(3):
        description = get_image_description(openai, data, system_prompt=system_prompt, user_prompt=user_prompt, is_image_type=is_image_type)
        descriptions.append(description)
        
    # Convert descriptions to JSON for confidence analysis
    descriptions_str = json.dumps(descriptions)
    confidence_result = get_confidence_level(openai, descriptions_str)
    
    # Show results in Streamlit UI
    # st.write("Descriptions:")
    # st.write(descriptions)
    
    st.write("Confidence Levels:")
    st.write(confidence_result)

    return descriptions, confidence_result