import os
import sys

#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2"

#os.environ["PIP_CACHE_DIR"]="/gpfs/projects/CascanteBonillaGroup/jmurrugarral/anaconda3/.cache"
#os.environ["HF_HOME"]="/gpfs/projects/CascanteBonillaGroup/jmurrugarral/anaconda3/.cache"
#os.environ["HF_DATASETS_CACHE"]="/gpfs/projects/CascanteBonillaGroup/jmurrugarral/anaconda3/.cache/datasets"
#os.environ["TRANSFORMERS_CACHE"]="/gpfs/projects/CascanteBonillaGroup/jmurrugarral/anaconda3/.cache/models"


#__dir__ = "/gpfs/projects/CascanteBonillaGroup/paola/PaddleOCR"
#sys.path.append(__dir__)
#sys.path.insert(0, os.path.abspath(os.path.join(__dir__, "../..")))
#sys.path.append("/gpfs/projects/CascanteBonillaGroup/paola/EVF-SAM/")

from transformers import AutoProcessor
from vllm import LLM, SamplingParams
from qwen_vl_utils import process_vision_info
import torch

import re
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

class QWEN_VLM:

    def __init__(self, model, processor):
        
        self.llm = model
        self.processor = processor 
        self.sampling_params = SamplingParams(
            temperature=0.1,
            top_p=0.001,
            repetition_penalty=1.05,
            max_tokens=1024,
            stop_token_ids=[],
            logprobs=20,
        )     
 
        data_list = [['bank account,financial statement', 'na'], ['mail', 'address'], ['credit card,debit card', 'na'], ['billing statement,customer receipt', 'na'], ['na', 'medical information,patient medical record'], ['na', 'na'], ['loan,financial statement', 'credit history'], ['prescription', 'na'], ['na', 'na'], ['na', 'na'], ['biometric', 'na'], ['transcript', 'education level'], ['employment', 'id card, driver license'], ['na', 'medical information,patient medical record'], ['na', 'na'], ['medical information,patient medical record', 'na'], ['biometric', 'na'], ['na', 'na'], ['phone number', 'na']]
        flattened_data = self.flatten_list_remove_na(data_list)
        all_categories = flattened_data
        unique_categories = list(set(all_categories))
        
        processed_strs = []
        for s in unique_categories:
            processed_strs.extend(self.process_string(s))
        
        unique_categories = []
        unique_categories = processed_strs
        unique_categories.append("credit card number")
        unique_categories.append("personal name")
        unique_categories.append("company name")
        unique_categories.append("unsure")
        unique_categories.append("other")
        
        self.unique_categories = unique_categories

        with open(f'/gpfs/projects/CascanteBonillaGroup/paola/Qwen2.5-VL/label2item_list.json', 'r') as file:
            self.unique_categories_per_metacategory = json.load(file)


    def flatten_list_remove_na(self, data_list):
        flat_list = []
        for sublist in data_list:
            for item in sublist:
                if item.lower() != 'na':
                    flat_list.append(item)
    
        return flat_list

    def process_string(self, input_string):
        if ',' in input_string:
            parts = input_string.split(',')
            cleaned_parts = [part.strip() for part in parts if part.strip()]
        else:
            cleaned_parts = [input_string]
    
        return cleaned_parts
    
    def guided_labeling(self, image, strs):
    
        messages = [
            {"role": "system", "content": """You are a helpful assistant. Output json {'text': text, 'category': category}"""},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image, # pil_src_im,
                    },
                    # {"type": "text", "text": f"Name Entity Recognition this text: {', '.join(strs)}. Output the label for each word in json format. Follow these categories {unique_categories}. Include all words."},
                    {"type": "text", "text": f"Name Entity Recognition each word in this list: {strs} using these categories: {self.unique_categories}. Use the image as context."},
                    # {"type": "text", "text": f"Based on the image, label each word in this sentence '{' '.join(strs)}' using these categories: {unique_categories}."},
                    # {"type": "text", "text": f"The text {'898'} appears in the image? Does it refers to: {' ,'.join(unique_categories)}?"},
                    # {"type": "text", "text": f"The text {'o1'} appears in the image? Does it refers to: {' ,'.join(unique_categories)}? Therefore, it refers to ..."},
                    # {"type": "text", "text": f"Does the image content related to one of this categories: {' ,'.join(self.unique_categories)}? Reason and output Therefore, the category is: ... Limit your response to the most probable category"},
                ],
            }
        ]
        
        # processor = AutoProcessor.from_pretrained(MODEL_PATH)
        prompt = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        image_inputs, video_inputs = process_vision_info(messages)

        mm_data = {}
        if image_inputs is not None:
            mm_data["image"] = image_inputs
        if video_inputs is not None:
            mm_data["video"] = video_inputs

        llm_inputs = {
            "prompt": prompt,
            "multi_modal_data": mm_data,
        }

        outputs = self.llm.generate([llm_inputs], sampling_params=self.sampling_params)
        generated_text = outputs[0].outputs[0].text
        return generated_text

    def _guided_labeling(self, image):
    
        messages = [
            # {"role": "system", "content": """You are a helpful assistant."""},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image, # pil_src_im,
                    },
                    # {"type": "text", "text": f"Name Entity Recognition this text: {', '.join(strs)}. Output the label for each word in json format. Follow these categories {unique_categories}. Include all words."},
                    # {"type": "text", "text": f"Name Entity Recognition each word in this list: {strs} using these categories: {unique_categories}. Use the image as context."},
                    # {"type": "text", "text": f"Based on the image, label each word in this sentence '{' '.join(strs)}' using these categories: {unique_categories}."},
                    # {"type": "text", "text": f"The text {'898'} appears in the image? Does it refers to: {' ,'.join(unique_categories)}?"},
                    # {"type": "text", "text": f"The text {'o1'} appears in the image? Does it refers to: {' ,'.join(unique_categories)}? Therefore, it refers to ..."},
                    {"type": "text", "text": f"Does the image content related to one of this categories: {' ,'.join(self.unique_categories)}? Reason and output Therefore, the category is: ... Limit your response to the most probable category"},
                ],
            }
        ]
        
        # processor = AutoProcessor.from_pretrained(MODEL_PATH)
        prompt = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        image_inputs, video_inputs = process_vision_info(messages)

        mm_data = {}
        if image_inputs is not None:
            mm_data["image"] = image_inputs
        if video_inputs is not None:
            mm_data["video"] = video_inputs

        llm_inputs = {
            "prompt": prompt,
            "multi_modal_data": mm_data,
        }

        outputs = self.llm.generate([llm_inputs], sampling_params=self.sampling_params)
        generated_text = outputs[0].outputs[0].text
        return generated_text
    
    def call_vlm(self, img_full_path, prompt, temperature = 1, top_p = 0.8, grounding = True, max_token = 128):
    
        # max_new_tokens = max_token
        messages = [
            # {
            #     "role": "system",
            #     "content": "You are a helpful assistant"
            # },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    },
                    {
                      "image": img_full_path,
                    #   "min_pixels": 1000 * 1000,
                    #   "max_pixels": 1000 * 1000,
                    }
                ]
            }
        ]

        # processor = AutoProcessor.from_pretrained(MODEL_PATH)
        prompt = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        image_inputs, video_inputs = process_vision_info(messages)

        mm_data = {}
        if image_inputs is not None:
            mm_data["image"] = image_inputs
        if video_inputs is not None:
            mm_data["video"] = video_inputs

        llm_inputs = {
            "prompt": prompt,
            "multi_modal_data": mm_data,
        }

        outputs = self.llm.generate([llm_inputs], sampling_params=self.sampling_params)
        response = outputs[0].outputs[0].text
    
        return response, image_inputs[0]
    
    def extract_bbox(self, data):
        """
        Extracts the bbox_2d array from a given JSON-like string.

        :param data: JSON-like string containing bbox_2d
        :return: Extracted bbox_2d as a list or None if not found
        """
        # Extract JSON part using regex
        match = re.search(r'```json\n(.*?)\n```', data, re.DOTALL)

        if match:
            try:
                json_data = json.loads(match.group(1).strip())  # Parse JSON after stripping spaces

                # Case 1: JSON is a list
                if isinstance(json_data, list) and json_data and "bbox_2d" in json_data[0]:
                    return json_data[0]["bbox_2d"]

                # Case 2: JSON is a dictionary
                if isinstance(json_data, dict) and "bbox_2d" in json_data:
                    return json_data["bbox_2d"]

            except json.JSONDecodeError:
                return None  # Handle JSON parsing errors

        return None  # Return None if no valid bbox_2d found    
    
    def crop_image(self, image_path, bbox):
        """
        Crop an image using the given bounding box and save the result.

        :param image_path: Path to the input image
        :param bbox: Bounding box in [x_min, y_min, x_max, y_max] format
        :param output_path: Path to save the cropped image
        """
        # Open the image
        image = Image.open(image_path)

        # Crop the image using the bbox
        cropped_image = image.crop(bbox)

        return cropped_image
    
    def detect_rotation(self, cropped_image, plausible_rotations):
        
        yes_probs = []
    
        for rotation_ii in plausible_rotations:
            cropped_image_tmp = cropped_image.copy()
            rotated_image = cropped_image_tmp.rotate(rotation_ii)
    
            # _, rotated_image = self.rotate_image(pil_to_opencv(cropped_image_tmp), rotation_ii, 'rotated.jpg')
    
            #plt.imshow(rotated_image)
            #plt.show()
            
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": rotated_image,
                        },
                    {"type": "text", "text": "Is this document properly aligned? Answer yes or no."},
                ],
                }
            ]

            # processor = AutoProcessor.from_pretrained(MODEL_PATH)
            prompt = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            image_inputs, video_inputs = process_vision_info(messages)

            mm_data = {}
            if image_inputs is not None:
                mm_data["image"] = image_inputs
            if video_inputs is not None:
                mm_data["video"] = video_inputs

            llm_inputs = {
                "prompt": prompt,
                "multi_modal_data": mm_data,
            }

            outputs = self.llm.generate([llm_inputs], sampling_params=self.sampling_params)
            # response = outputs[0].outputs[0].text
            yes_probs.append(outputs[0].outputs[0].logprobs[0][9454].logprob)

        # get two max scores:
        max_idxs = sorted(range(len(yes_probs)), key=lambda i: yes_probs[i], reverse=True)[:5]
    
        # return plausible_rotations[max_idxs[0]], plausible_rotations[max_idxs[1]], yes_probs
        return plausible_rotations[max_idxs], max_idxs, yes_probs
    
    def rotate_image(self, image, angle, output_path):
    
        # Get image dimensions
        (h, w) = image.shape[:2]
    
        # Compute the center of the image
        center = (w // 2, h // 2)
    
        # Get the rotation matrix
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
    
        # Compute the new bounding dimensions
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])
        new_w = int((h * sin) + (w * cos))
        new_h = int((h * cos) + (w * sin))
    
        # Adjust the rotation matrix for translation
        M[0, 2] += (new_w / 2) - center[0]
        M[1, 2] += (new_h / 2) - center[1]
    
        if angle == 0:
            rotated = image
        elif angle == 90:
            rotated = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        elif angle == 180:
            rotated = cv2.rotate(image, cv2.ROTATE_180)
        elif angle == 270:
            rotated = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        else:
            rotated = cv2.warpAffine(image, M, (new_w, new_h))
    
        M   = np.append(M, [[0,0,1]],axis=0)
        M_I = np.linalg.inv(M).astype(np.float32)
        
        # Save the rotated image
        cv2.imwrite(output_path, rotated)
    
        return M_I, opencv_to_pil(rotated.astype(np.uint8))


    def get_finegrained_text(self,rotated_image):
        sampling_params = SamplingParams(
            temperature=0.1,
            top_p=0.001,
            repetition_penalty=1.05,
            max_tokens=5000,
            stop_token_ids=[],
            logprobs=20,
        )
    
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": rotated_image,
                        # "min_pixels": 224 * 224,
                        # "max_pixels": 1280 * 28 * 28,
                    },
                    {"type": "text", "text": "Locate all text (bbox coordinates). Include all readable and blury text"},
                ],
            },
        ]
    
        # processor = AutoProcessor.from_pretrained(MODEL_PATH)
        prompt = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        image_inputs, video_inputs = process_vision_info(messages)
    
        mm_data = {}
        if image_inputs is not None:
            mm_data["image"] = image_inputs
        if video_inputs is not None:
            mm_data["video"] = video_inputs
    
        llm_inputs = {
            "prompt": prompt,
            "multi_modal_data": mm_data,
        }
    
        outputs = self.llm.generate([llm_inputs], sampling_params=sampling_params)
        generated_text = outputs[0].outputs[0].text
    
        return generated_text

    def guided_labeling_per_match(self, image, strs):
    
        messages = [
            {"role": "system", "content": """You are a helpful assistant."""},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image, # pil_src_im,
                    },
                    # {"type": "text", "text": f"Name Entity Recognition this text: {', '.join(strs)}. Output the label for each word in json format. Follow these categories {unique_categories}. Include all words."},
                    {"type": "text", "text": f"Based on the image, classify this text: '{strs} 'using these categories: {self.unique_categories}. Output only one category."},
                    # {"type": "text", "text": f"Based on the image, label each word in this sentence '{' '.join(strs)}' using these categories: {unique_categories}."},
                    # {"type": "text", "text": f"The text {'898'} appears in the image? Does it refers to: {' ,'.join(unique_categories)}?"},
                    # {"type": "text", "text": f"The text {'o1'} appears in the image? Does it refers to: {' ,'.join(unique_categories)}? Therefore, it refers to ..."},
                    # {"type": "text", "text": f"Does the image content related to one of this categories: {' ,'.join(self.unique_categories)}? Reason and output Therefore, the category is: ... Limit your response to the most probable category"},
                ],
            }
        ]
        
        # processor = AutoProcessor.from_pretrained(MODEL_PATH)
        prompt = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        image_inputs, video_inputs = process_vision_info(messages)

        mm_data = {}
        if image_inputs is not None:
            mm_data["image"] = image_inputs
        if video_inputs is not None:
            mm_data["video"] = video_inputs

        llm_inputs = {
            "prompt": prompt,
            "multi_modal_data": mm_data,
        }

        outputs = llm.generate([llm_inputs], sampling_params=self.sampling_params)
        generated_text = outputs[0].outputs[0].text
        return generated_text
    

    def guided_labeling_per_match_per_metacategory(self, image, strs, metacategory):
    
        unique_categories = self.unique_categories_per_metacategory[metacategory]["contained_info"]
        unique_categories.append("other")
        unique_categories.append("none")

        messages = [
            {"role": "system", "content": """You are a helpful assistant."""},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image, # pil_src_im,
                    },
                    # {"type": "text", "text": f"Name Entity Recognition this text: {', '.join(strs)}. Output the label for each word in json format. Follow these categories {unique_categories}. Include all words."},
                    {"type": "text", "text": f"Based on the image, classify this text: '{strs} 'using these categories: {unique_categories}. Output only one category."},
                    # {"type": "text", "text": f"Based on the image, label each word in this sentence '{' '.join(strs)}' using these categories: {unique_categories}."},
                    # {"type": "text", "text": f"The text {'898'} appears in the image? Does it refers to: {' ,'.join(unique_categories)}?"},
                    # {"type": "text", "text": f"The text {'o1'} appears in the image? Does it refers to: {' ,'.join(unique_categories)}? Therefore, it refers to ..."},
                    # {"type": "text", "text": f"Does the image content related to one of this categories: {' ,'.join(self.unique_categories)}? Reason and output Therefore, the category is: ... Limit your response to the most probable category"},
                ],
            }
        ]
        
        # processor = AutoProcessor.from_pretrained(MODEL_PATH)
        prompt = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        image_inputs, video_inputs = process_vision_info(messages)

        mm_data = {}
        if image_inputs is not None:
            mm_data["image"] = image_inputs
        if video_inputs is not None:
            mm_data["video"] = video_inputs

        llm_inputs = {
            "prompt": prompt,
            "multi_modal_data": mm_data,
        }

        outputs = llm.generate([llm_inputs], sampling_params=self.sampling_params)
        generated_text = outputs[0].outputs[0].text
        return generated_text
