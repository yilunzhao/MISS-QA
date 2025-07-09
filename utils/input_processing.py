import base64
from utils.constant import COT_PROMPT
import requests
import os
import time
from tqdm import tqdm
import hashlib
import json
import base64
from PIL import Image
from io import BytesIO

def prepare_context(paper_path,section_list):
    with open(paper_path,'r',encoding='utf-8')as f:
        paper = json.load(f)
    if len(section_list)==0:
        context = ""
        for sec in paper['sections']:
            context += sec['section_name'] + ':\n' + sec['text'] + '\n'
    else:
        top_sections = []
        section_list = list(set(section_list))
        for item in section_list:
            if item.split('.')[0] not in top_sections:
                top_sections.append(item.split('.')[0])
        context = ""
        for sec in paper['sections']:
            if sec['section_id'].split('.')[0] in top_sections:
                context += sec['section_name'] + ':\n' + sec['text'] + '\n'
    return context


def prepare_qa_text_input(model_name, query, prompt):
    """
    Prepare the text part of QA prompt based on the question type.
    - If question_type is "multiple-choice", it will substitute the question and the formatted choice list.
    - If question_type is "open-ended", it will substitute only the question.
    """
    mask_number = query['masked_number']
    user_prompt = prompt[mask_number]
    question = query['masked_question']
    paper_path = query['paper_path']
    section_list = query['relevant_section_ids']
    context = prepare_context(paper_path,section_list)
    caption = query['caption']
    qa_text_prompt = user_prompt.substitute(question=question,context=context,caption=caption)
    # Return the formatted text message object and raw string
    qa_text_message = {
        "type": "text",
        "text": qa_text_prompt
    }
    return qa_text_message, qa_text_prompt

def prepare_single_image_input(model_name, image_source):
    """
    Prepare a single image input for different model requirements.

    This function now accepts either:
      - A local image file path (e.g., "path/to/image.jpg")
      - An image URL (e.g., "http://.../some_image.jpg" or "https://...")
    
    It then converts the image content into base64 internally before returning
    the final structure, which varies depending on 'model_name':
      - a dict with {"type": "image", "source": {...}} for Claude
      - a raw base64 string for vLLM or other HF-based models
      - a dict with {"type": "image_url", ...} otherwise

    Args:
        model_name (str): The name of the model. Used to decide the returned format.
        image_source (str): Either a local file path or an image URL.

    Returns:
        A dictionary or a base64 string, depending on the model_name.
    """

    # 1. Determine if the source is a local path or a URL
    if image_source.startswith("http://") or image_source.startswith("https://"):
        # If it's an HTTP/HTTPS URL, download the image
        response = requests.get(image_source)
        response.raise_for_status()
        image_content = response.content
    else:
        # Otherwise, assume it's a local file path
        if not os.path.exists(image_source):
            raise FileNotFoundError(f"Image file not found: {image_source}")
        with open(image_source, "rb") as f:
            image_content = f.read()

    # 2. Convert the image content to base64
    image_base64 = base64.b64encode(image_content).decode("utf-8")

    # 3. Return the structure based on the model_name
    if "claude" in model_name:
        # Claude usually requires {"type":"image", "source":{...}}
        return {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/jpeg",
                "data": image_base64,
            },
        }
    elif "/" in model_name:
        # For vLLM or other HF-based models, return raw base64
        return image_base64
    else:
        # Some fallback or custom format
        return {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{image_base64}",
            },
        }


def prepare_qa_inputs(model_name, queries, prompt=COT_PROMPT):
    """
    Prepare the final list of messages that combine single image input with QA text.
    - Each query is turned into a list containing a user role with two items: image + text QA.
    """
    messages = []
    for query in tqdm(queries):
        # Create the QA text portion
        qa_text_message, qa_text_prompt = prepare_qa_text_input(model_name, query, prompt)

        # Create or retrieve the single image data structure
        single_image_data = prepare_single_image_input(model_name, query['figure_path'])

        # Combine them into a "user" content list
        prompt_message = [
            {
                "role": "user",
                "content": [
                    single_image_data,
                    qa_text_message
                ],
            }
        ]
        messages.append(prompt_message)
    return messages
