# --- Security Hardening: SSRF Guard (RFC 1918 & Cloud Metadata) ---
import socket
import ipaddress
from urllib.parse import urlparse

def is_safe_url(target_url: str) -> bool:
    """Validates that target_url resolves strictly to public, routable IP addresses."""
    try:
        parsed = urlparse(target_url)
        hostname = parsed.hostname
        if not hostname:
            return False
        addr_info = socket.getaddrinfo(hostname, None)
        for entry in addr_info:
            ip = ipaddress.ip_address(entry[4][0])
            if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or not ip.is_global:
                return False
        return True
    except Exception:
        return False
# ------------------------------------------------------------------

import os
import sys
import base64
import io
import requests
from PIL import Image
from typing import Annotated, Literal, List, Dict, Optional, Tuple, Any
import numpy as np
import random
import torch
import pandas as pd
import json

from autogen.agentchat.contrib.img_utils import get_pil_image, get_image_data, pil_to_data_uri
from langchain.tools import BaseTool
from huggingface_hub import login, HfApi, delete_repo
from omegaconf import OmegaConf

def get_model_zoo() -> Dict[str, List[str]]:
    '''
    Returns:
        dict: A dictionary with keys representing computer vision task types and values as lists of checkpoint names.
              Example format:
              {
                  'instance-segmentation': ['checkpoint1', 'checkpoint2'],
                  'image-classification': ['checkpoint3'],
                  'image-regression': [],
              }
    Naming rule for each checkpoint: {plant-species}_{plant-task}_{(optional) training-dataset}_{model-name}_{(optional) finetuning-method}
    '''
    with open("./model_zoo.json", "r") as f:
        model_zoo = json.load(f)
    return model_zoo

def update_model_zoo(task_type: Annotated[str, "The type of computer vision task."],
                     model_name: Annotated[str, "The name of the model checkpoint."]):
    with open("./model_zoo.json", "r") as f:
        model_zoo = json.load(f)
    model_zoo[task_type].append(model_name)
    with open("model_zoo.json", "w") as f:
        json.dump(model_zoo, f)
    print("Model added to model_zoo successfully!")

def set_env_vars(env_file: str):
    env_conf = OmegaConf.load(env_file)
    for k, v in env_conf.items():
        os.environ[k] = str(v)

def set_random_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Random seed set to: {seed}")

def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}")

def handle_grayscale_image(image):
    np_image = np.array(image)
    if np_image.ndim == 2:
        tiled_image = np.tile(np.expand_dims(np_image, -1), 3)
        return Image.fromarray(tiled_image)
    else:
        return Image.fromarray(np_image)

def download_hffile(url, headers):
    if not is_safe_url(url):
        raise ValueError(f"SSRF Protection: Refusing to fetch restricted or private IP address for URL: {url}")
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        return response.json()  # Load JSON into memory
    else:
        print(f"Failed to download {url}. Status code: {response.status_code}")
        return None

def delete_hfdata(data_list: List[str],
                  data_type: str =  'model', # or dataset
                #   repo_name: str = 'fengchen025',
                  ):
    for data_name in data_list:
        delete_repo(repo_id=f'{data_name}', repo_type=data_type)

def make_dir(dir_path: Annotated[str, "The path to the directory to be created."]) -> str:
    if not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)
        return f"Directory '{dir_path}' created successfully."
    else:
        return f"Directory '{dir_path}' already exists."

def extract_column_name_from_csv(file_path: Annotated[str, "The path to the CSV file."]) -> str:
    df = pd.read_csv(file_path)
    return df.columns.to_list()

# img loading
def load_images(image_paths):
    # support local and request.get
    images = []
    for path in image_paths:
        if os.path.isfile(path):  # Check if it's a local file path
            images.append((Image.open(path)).convert("RGB"))
        else:  # Otherwise, assume it's a URL
            response = requests.get(path, stream=True)
            images.append(Image.open(response.raw))
    return images

def calculator(a: int, 
               b: int, 
               operator: Annotated[str, "operator"]) -> int:
    """
    A calculator performing basic arithmetic operations between two integers.

    Args:
        a (int): The first integer.
        b (int): The second integer.
        operator (str): The arithmetic operation to perform. 
                        Can be one of '+', '-', '*', '/'.

    Returns:
        int: The result of the arithmetic operation.
    """
    if operator == "+":
        return a + b
    elif operator == "-":
        return a - b
    elif operator == "*":
        return a * b
    elif operator == "/":
        return int(a / b)
    else:
        raise ValueError("Invalid operator")