import base64
import json  # Don't forget to import the json module
import openai

def get_image_description(client, uploaded_file, system_prompt, user_prompt, is_image_type=False):
    # Ensure that uploaded_file is always treated as a list
    if not isinstance(uploaded_file, list):
        uploaded_file = [uploaded_file]  # Convert single item to a list

    # Encode the content based on the type of the file
    if is_image_type:
        encoded_content_list = [get_image_content(base64.b64encode(i).decode('utf-8')) for i in uploaded_file]  # Handle image bytes/memoryview
    else:
        encoded_content_list = [get_text_content(i) for i in uploaded_file]  # Handle text strings

    try:
        # Create the GPT-4 API request
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": system_prompt
                        },
                    ]
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"Please respond specifically to the user's prompt in JSON format: {user_prompt}."
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        *encoded_content_list
                    ],
                }
            ],
            max_tokens=1500,
            temperature=0.3,
            response_format={"type": "json_object"}  # Turn on JSON mode
        )

        # Attempt to parse the response as JSON
        json_response = json.loads(response.choices[0].message.content)
        return json_response  # Return the JSON response
    except openai.OpenAIError as e:
        raise RuntimeError(f"An error occurred with the OpenAI API: {e}")
    except json.JSONDecodeError:
        raise ValueError("The model response is not a valid JSON.")
    except openai.AuthenticationError:
        raise ValueError("Invalid API key. Please check your OpenAI API key.")
    except openai.RateLimitError:
        raise RuntimeError("Rate limit exceeded. Please try again later.")
    except openai.APIConnectionError:
        raise RuntimeError("Network connection error. Please check your internet connection.")
    except Exception as e:
        raise RuntimeError(f"An unexpected error occurred: {e}")


def get_image_content(image_content):
    return {
        "type": "image_url",
        "image_url": {"url": f"data:image/png;base64,{image_content}"}
    }

def get_text_content(text_content):
    return {
        "type": "text",
        "text": text_content
    }
