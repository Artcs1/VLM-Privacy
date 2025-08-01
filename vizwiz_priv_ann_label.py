import os
import sys

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3"

from transformers import AutoProcessor
from vllm import LLM, SamplingParams
from qwen_vl_utils import process_vision_info
import torch

from tqdm import tqdm

# Initialize the processor
#processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-72B-Instruct")
processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-72B-Instruct")
    
# Initialize the LLM
# from vllm.config import PoolerConfig
#llm = LLM(model="Qwen/Qwen2.5-VL-72B-Instruct", tensor_parallel_size=4, dtype=torch.bfloat16) # , task="generate", override_pooler_config=PoolerConfig(pooling_type="ALL"))
llm = LLM(model="Qwen/Qwen2.5-VL-72B-Instruct", tensor_parallel_size=4, dtype=torch.bfloat16) # , task="generate", override_pooler_config=PoolerConfig(pooling_type="ALL"))
    
# llm = LLM(model="Qwen/Qwen2.5-VL-72B-Instruct", torch_dtype=torch.bfloat16,
#         attn_implementation="flash_attention_2",
#         device_map="auto",
#     )
    


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
#from model.segment_anything.utils.transforms import ResizeLongestSide
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


from agents.qwen import QWEN_VLM
from agents.orientation import ORIENTATION_AGENT
from agents.segmentation import SEGMENTATION_AGENT
from agents.ocr import OCR_AGENT, pladdleOCR
from utils import *

from tqdm import tqdm

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

def main(): 

    qwen = QWEN_VLM(llm, processor)

    if not os.path.exists(f"results_qwen_72B_creditcards_labels/"):
        os.makedirs(f"results_qwen_72B_creditcards_labels/")
    
    if not os.path.exists(f"results_qwen_72B_labels/"):
        os.makedirs(f"results_qwen_72B_labels/")

    paths = ["results_qwen_72B", "results_qwen_72B_creditcards"]

    for folder_path in paths:
    
        files = glob.glob(folder_path + "/*")
          
        for ii in tqdm(range(0, len(files))):
    
            filepath = files[ii]
            with open(filepath, 'r') as file:
                data = json.load(file)
        
            img = data['image_path']
            try:
                str = [text_content_tmp['text_content'] for text_content_tmp in data['detailed_data'][0]['data_vlm']]
                labels_full_image = qwen.guided_labeling(img, str)
        
                masked_image_cv = base64_str_to_image2(data['masked_image'])
                masked_image_pil = opencv_to_pil(masked_image_cv)
                labels_masked_image = qwen.guided_labeling(masked_image_pil, str)
        
                rotated_image_cv = base64_str_to_image2(data['detailed_data'][0]['rotated_image'])
                rotated_image_pil = opencv_to_pil(rotated_image_cv)
                labels_rotated_image = qwen.guided_labeling(rotated_image_pil, str)
        
                all_labels = {'image_path': filepath, 
                              'labels_full_image': extract_bbox_removing_incomplete(labels_full_image), 
                              'labels_masked_image': extract_bbox_removing_incomplete(labels_masked_image), 
                              'labels_rotated_image': extract_bbox_removing_incomplete(labels_rotated_image)}
            except:
                all_labels = {'image_path': filepath, 
                             'labels_full_image': "Empty", 
                             'labels_masked_image': "Empty", 
                             'labels_rotated_image': "Empty"}
        
            filename = os.path.basename(filepath)
            if 'creditcards' in folder_path:    
                with open(f"results_qwen_72B_creditcards_labels/{filename}", "w") as fp:
                    json.dump(all_labels, fp, indent=4)
            else:
                with open(f"results_qwen_72B_labels/{filename}", "w") as fp:
                    json.dump(all_labels, fp, indent=4)
        
    
if __name__ == '__main__':
    main() 
