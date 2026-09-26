import os
import json
import re
import base64
from pathlib import Path
from openai import OpenAI
from PIL import Image
import numpy as np
import torch
from colorama import Fore, Style, init

# === Config ===

SYSTEM_PROMPT = "You are a helpful visual assistant."

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

def encode_image_base64(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")
    
def transform_data_format(list:list) -> str:
    
    input = []

    for item in list:
        dict = {}
        if 'image' in item and isinstance(item['image'], str):
            image = encode_image_base64(item['image'])
            dict["type"] = "image_url"
            dict["image_url"] = {"url": f"data:image/png;base64,{image}"}
        elif 'text' in item:
            dict["type"] = "text"
            dict["text"] = item['text']
        input.append(dict)

    return input  

def generate_content(model_name, content):

    input = transform_data_format(content)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": input}
    ]

    print(f"{Fore.YELLOW} Generating content with model {Style.RESET_ALL} {model_name}...")
    # print(f"{Fore.YELLOW} Input text data: {Style.RESET_ALL}", input)

    completion = client.chat.completions.create(model=model_name, messages=messages)
    print(f"{Fore.YELLOW} Received response from {model_name} {Style.RESET_ALL}:", f"{completion.choices[0].message.content.strip()}")
    return completion.choices[0].message.content.strip()