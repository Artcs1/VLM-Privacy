import os
import sys

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2"

os.environ["PIP_CACHE_DIR"]="/gpfs/projects/CascanteBonillaGroup/jmurrugarral/anaconda3/.cache"
os.environ["HF_HOME"]="/gpfs/projects/CascanteBonillaGroup/jmurrugarral/anaconda3/.cache"
os.environ["HF_DATASETS_CACHE"]="/gpfs/projects/CascanteBonillaGroup/jmurrugarral/anaconda3/.cache/datasets"
os.environ["TRANSFORMERS_CACHE"]="/gpfs/projects/CascanteBonillaGroup/jmurrugarral/anaconda3/.cache/models"


__dir__ = "/gpfs/projects/CascanteBonillaGroup/paola/PaddleOCR"
sys.path.append(__dir__)
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, "../..")))
sys.path.append("/gpfs/projects/CascanteBonillaGroup/paola/EVF-SAM/")

from transformers import AutoProcessor
from vllm import LLM, SamplingParams
from qwen_vl_utils import process_vision_info
import torch

import re
import pickle
import html
import argparse
import glob
import time
import json
import torch
import random
import io
import ast
import cv2
import torch.nn.functional as F
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
import tools.infer.utility as utility
import pandas as pd
import base64
import numpy as np
from tqdm import tqdm
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont, ImageColor
from IPython.display import display   
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoTokenizer, BitsAndBytesConfig
from model.segment_anything.utils.transforms import ResizeLongestSide
from shapely.validation import explain_validity
from shapely.geometry import Polygon
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
from collections import Counter
from ppocr.utils.logging import get_logger
from ppocr.utils.utility import get_image_file_list, check_and_read
from ppocr.data import create_operators, transform
from ppocr.postprocess import build_post_process
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info


additional_colors = [colorname for (colorname, colorcode) in ImageColor.colormap.items()]

def qwen25():
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-VL-7B-Instruct",
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
    model.eval()
    return model, processor

def llava16():
    from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
    llava_processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf") 
    llava_model = LlavaNextForConditionalGeneration.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf", torch_dtype=torch.float16, device_map="auto") #, low_cpu_mem_usage=True) 
    llava_model.generation_config.pad_token_id = llava_model.generation_config.eos_token_id
    return llava_model, llava_processor

def gemma3():
    from transformers import AutoProcessor, Gemma3ForConditionalGeneration
    ckpt = "google/gemma-3-4b-it"
    model = Gemma3ForConditionalGeneration.from_pretrained(
        ckpt, device_map="auto", torch_dtype=torch.bfloat16,
    )
    processor = AutoProcessor.from_pretrained(ckpt)
    return model, processor

def vqa_yes_prob(model_7b, processor_7b, image, question_prompt, model="qwen25", device="cuda"):

    if model == "qwen25":
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image,
                        "min_pixels": 224 * 224,
                        # "max_pixels": 1280 * 28 * 28,
                    },
                {"type": "text", "text": f"{question_prompt}? Answer yes or no."},
            ],
            }
        ]

        # Preparation for inference
        text = processor_7b.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor_7b(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(model_7b.device)

        # Inference: Generation of the output
        with torch.no_grad():
            outputs = model_7b.generate(**inputs, max_new_tokens=128,
                                    return_dict_in_generate=True,
                                    use_cache=False,
                                    # return_dict=True,
                                    output_scores=True,
                                    do_sample=False)
            generated_ids = outputs["sequences"]
            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = processor_7b.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )

            inputs_for_prob = processor_7b(text=["Yes"],
                        images=image_inputs,
                        videos=video_inputs,
                        padding=True,
                        return_tensors="pt")
            yes_id = inputs_for_prob['input_ids'][0][0].item()


            inputs_for_prob = processor_7b(text=["No"],
                            images=image_inputs,
                            videos=video_inputs,
                            padding=True,
                            return_tensors="pt")
            no_id = inputs_for_prob['input_ids'][0][0].item()

            logits = outputs["scores"][0][0]
            probs = (torch.nn.functional.softmax(torch.tensor([logits[yes_id], logits[no_id]]), dim=0).detach().cpu().numpy())
            # print(f"Output text: {output_text} :: {rotation_ii} - Yes prob: {probs[0]}, No prob: {probs[1]}")
            
            return probs[0], output_text #return yes prob
        
    elif model == "llava16":
        conversations = []
        # for text in captions:
        conversations.append(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"{question_prompt}? Answer yes or no."},
                        {"type": "image"},
                    ],
                },
            ]
        )
            
        prompt = processor_7b.apply_chat_template(conversations, add_generation_prompt=True)
        inputs = processor_7b(images=image, text=prompt, padding=True, return_tensors="pt").to(device, torch.float16)

        # autoregressively complete prompt
        with torch.no_grad():
            output = model_7b.generate(**inputs, max_new_tokens=200,
                                    return_dict_in_generate=True,
                                    output_scores=True,)
            
        labels_for_prob = processor_7b(images=image, text="yes no", padding=True, return_tensors="pt").to(device, torch.float16)
        yes_id = labels_for_prob['input_ids'][0][1].item()
        no_id = labels_for_prob['input_ids'][0][2].item()

        all_probs = []
        for idx in range(output['scores'][0].shape[0]):
            logits = output.scores[0][idx]

            probs = (torch.nn.functional.softmax(torch.tensor([logits[yes_id], logits[no_id]]), dim=0).detach().cpu().numpy())

            all_probs.append(probs[0])

        return all_probs
    
    elif model == "gemma3":
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": "You are a helpful assistant."}]
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": f"{question_prompt} Answer yes or no."}
                ]
            }
        ]

        inputs = processor_7b.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt"
        ).to(model_7b.device, dtype=torch.bfloat16)

        inputs_for_prob = processor_7b(text=["Yes"],
                    # images=image_inputs,
                    # videos=video_inputs,
                    padding=True,
                    return_tensors="pt")
        yes_id = inputs_for_prob['input_ids'][0][1].item()

        inputs_for_prob = processor_7b(text=["No"],
                        # images=image_inputs,
                        # videos=video_inputs,
                        padding=True,
                        return_tensors="pt")
        no_id = inputs_for_prob['input_ids'][0][1].item()

        input_len = inputs["input_ids"].shape[-1]

        with torch.inference_mode():
            generation = model_7b.generate(**inputs, max_new_tokens=100, do_sample=False,
                                        return_dict_in_generate=True,
                                        use_cache=False,
                                        # return_dict=True,
                                        output_scores=True,)
            sequences = generation[0][0][input_len:]

        decoded = processor_7b.decode(sequences, skip_special_tokens=True)
        logits = generation[1][0][0]
        probs = (torch.nn.functional.softmax(torch.tensor([logits[yes_id], logits[no_id]]), dim=0).detach().cpu().numpy())
        return float(probs[0]), decoded


def extract_number(name):
    match = re.search(r'(\d+)', os.path.basename(name))
    if match:
        return int(match.group(1))
    return None

def full_image_zero_outside_mask(file_path, mask):
    """Sets pixel values outside the mask to 0."""

    image = Image.open(file_path)
    image = np.array(image)

    # Ensure image and mask have compatible shapes
    if image.shape[:2] != mask.shape:
        raise ValueError("Image and mask must have the same height and width.")

    # Set values outside mask to 0
    image[~mask] = 0

    return Image.fromarray(image)

def base64_str_to_image2(b64_str):
    # Remove header if present
    if "base64," in b64_str:
        b64_str = b64_str.split("base64,")[-1]
    image_data = base64.b64decode(b64_str)
    image = Image.open(io.BytesIO(image_data))
    # Convert PIL image (RGB) to OpenCV format (BGR)
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.generic):
            return obj.item()  # Convert numpy scalar to Python scalar
        return super().default(obj)

def pil_to_opencv(pil_img):
    """
    Convert a PIL Image to an OpenCV image (NumPy array).
    Handles RGB, RGBA, and L (grayscale).
    """
    # Ensure it's a PIL Image object
    if not isinstance(pil_img, Image.Image):
        raise TypeError("Input must be a PIL Image.")

    # Convert to NumPy
    np_img = np.array(pil_img)

    # Handle different PIL modes
    if pil_img.mode == "RGB":
        # PIL (RGB) -> OpenCV (BGR)
        return cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR)
    elif pil_img.mode == "RGBA":
        # PIL (RGBA) -> OpenCV (BGRA)
        return cv2.cvtColor(np_img, cv2.COLOR_RGBA2BGRA)
    elif pil_img.mode == "L":
        # PIL (grayscale) just becomes 2D array, no color channel swap needed
        return np_img
    else:
        # Fallback: convert PIL to RGB first
        rgb_img = pil_img.convert("RGB")
        np_img = np.array(rgb_img)
        return cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR)

def opencv_to_pil(cv_img):
    """
    Convert an OpenCV image (NumPy array, BGR/BGRA/Gray) back to a PIL Image.
    """
    if not isinstance(cv_img, np.ndarray):
        raise TypeError("Input must be a NumPy array (OpenCV image).")

    # Check shape to figure out color space
    if len(cv_img.shape) == 2:
        # Grayscale
        return Image.fromarray(cv_img)
    elif len(cv_img.shape) == 3:
        channels = cv_img.shape[2]
        if channels == 3:
            # OpenCV (BGR) -> PIL (RGB)
            return Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))
        elif channels == 4:
            # OpenCV (BGRA) -> PIL (RGBA)
            return Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGRA2RGBA))
    
    # Fallback: Convert to RGB
    return Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))


def set_zero_outside_mask(image, mask):
    """Sets pixel values outside the mask to 0."""

    # Ensure image and mask have compatible shapes
    if image.shape[:2] != mask.shape:
        raise ValueError("Image and mask must have the same height and width.")

    # Create a copy of the image to avoid modifying the original
    masked_image = image.copy()

    # Set values outside mask to 0
    masked_image[~mask] = 255

    return masked_image

def extract_bbox_removing_incomplete(text):
    """
    Parse truncated JSON that looks like an array of objects,
    discard any incomplete final object, and extract bbox_2d arrays.
    """

    # 1) Strip away the initial ```json plus everything before it
    #    and any trailing backticks or text.
    #    This regex grabs everything after ```json until the end,
    #    then removes any trailing ``` if present.
    match = re.search(r'```json\s*(.*)', text, re.DOTALL)
    if not match:
        return None
    json_str = match.group(1)
    # Remove any trailing triple backticks and beyond
    json_str = re.sub(r'```.*$', '', json_str, flags=re.DOTALL).strip()

    # 2) If the entire thing is wrapped in [...] at the top level, remove them.
    #    We'll parse object by object ourselves.
    #    - We'll do this only if it starts with '[' and ends with ']'.
    if json_str.startswith('[') and json_str.endswith(']'):
        # remove the first '[' and the last ']'
        json_str = json_str[1:-1].strip()

    # 3) Collect each *complete* top-level `{ ... }` object
    #    ignoring any trailing incomplete object.
    objects = []
    brace_stack = 0
    start_idx = None
    for i, ch in enumerate(json_str):
        if ch == '{':
            if brace_stack == 0:
                # Potential start of a new object
                start_idx = i
            brace_stack += 1
        elif ch == '}':
            brace_stack -= 1
            if brace_stack == 0 and start_idx is not None:
                # We found a complete object from start_idx to i
                obj_str = json_str[start_idx:i+1]
                objects.append(obj_str)
                start_idx = None
    
    # Now `objects` holds each fully closed `{...}`

    # 4) Build a valid JSON array from these objects
    #    For example: [{...},{...},...]
    if not objects:
        return None  # No complete objects found
    array_str = "[" + ",".join(objects) + "]"

    # 5) Parse the array
    try:
        data_list = json.loads(array_str)
    except json.JSONDecodeError:
        return None  # Something else is malformed

    return data_list 

def plot_text_inside_image(rotated_image_v2, data):
    cropped_image_tmp = rotated_image_v2.copy()
    draw = ImageDraw.Draw(cropped_image_tmp)

    # Try to load a TTF font; if not available, fallback to the default font

    try:
        font = ImageFont.truetype("arial.ttf", size=16)
    except IOError:
        font = ImageFont.load_default()

    # Iterate over each item, draw the bounding box and overlay the text
    for item in data:
        bbox = item["bbox_2d"]  # [x1, y1, x2, y2]
        text = item["text_content"]

        # Draw the rectangle using a red outline with a thickness of 2 pixels
        draw.rectangle(bbox, outline="red", width=2)

        # Position the text slightly above the bounding box (adjust offset as needed)
        text_position = (bbox[0], max(bbox[1] - 20, 0))
        draw.text(text_position, text, fill="blue", font=font)

    # Save and display the resulting image
    # output_path = 'output.jpg'
    # img.save(output_path)
    
    # cropped_image_tmp.show()
    plt.figure(figsize=(20, 20))
    plt.imshow(cropped_image_tmp)
    plt.axis('off')
    plt.show()        

def image_to_base64_str(img_array):
    # img_array is your NumPy array in BGR format from OpenCV
    # Encode as PNG in memory
    success, encoded_image = cv2.imencode('.png', img_array)
    if not success:
        raise ValueError("Could not encode image as PNG.")
    # Convert to base64 bytes, then decode as ASCII/UTF-8 to get a string
    b64_str = base64.b64encode(encoded_image).decode('utf-8')
    return b64_str

def base64_str_to_image(b64_str):
    # Revert from base64 string back to OpenCV image (NumPy array)
    image_data = base64.b64decode(b64_str)
    np_arr = np.frombuffer(image_data, np.uint8)
    img_array = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    return img_array

def convert_np(obj):
    """Recursively convert NumPy types in obj to native Python types."""
    if isinstance(obj, dict):
        return {k: convert_np(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_np(x) for x in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()  # or map convert_np over each element if deeply nested
    else:
        return obj  # leave everything else alone

def polygon_orientation(points):
    """
    Computes the orientation (in degrees) of a polygon via PCA on its vertices.
    
    :param points: N x 2 numpy array of (x, y) polygon vertices
    :return: Angle in degrees in the range (-180, +180]
    """
    # 1. Compute centroid
    centroid = np.mean(points, axis=0)
    
    # 2. Shift polygon to the origin based on centroid
    shifted = points - centroid
    
    # 3. Perform PCA: compute covariance, eigenvalues, eigenvectors
    cov = np.cov(shifted.T)
    eigvals, eigvecs = np.linalg.eig(cov)
    
    # The principal axis corresponds to the eigenvector with the largest eigenvalue
    principal_axis = eigvecs[:, np.argmax(eigvals)]
    
    # 4. Compute the angle with respect to the x-axis
    angle = np.arctan2(principal_axis[1], principal_axis[0])
    
    # Convert from radians to degrees
    angle_deg = np.degrees(angle)
    
    # For a more standard orientation in [-180, 180], you can do:
    if angle_deg > 180:
        angle_deg -= 360
    elif angle_deg <= -180:
        angle_deg += 360
    
    return angle_deg

def is_horizontal(points, tolerance_degrees=5):
    """
    Checks if the polygon is 'horizontal' within a given tolerance (in degrees).
    
    :param points: N x 2 numpy array of (x, y) polygon vertices
    :param tolerance_degrees: angle within which we consider the polygon horizontal
    :return: True if horizontal within the tolerance, False otherwise
    """
    angle_deg = polygon_orientation(points)
    # A polygon can be near 0° or near 180° and still be "horizontal"
    # So we check absolute distance to 0 or 180
    angle_mod_180 = abs((angle_deg + 180) % 180 - 90)  # a way to measure how close to horizontal
    return angle_mod_180 >= (90 - tolerance_degrees)  

