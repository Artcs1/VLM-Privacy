import os
import sys

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3"

ANACONDA_PATH = "/gpfs/projects/CascanteBonillaGroup/jmurrugarral/anaconda3/"

os.environ["PIP_CACHE_DIR"]= ANACONDA_PATH +".cache"
os.environ["HF_HOME"]=ANACONDA_PATH + ".cache"
os.environ["HF_DATASETS_CACHE"]=ANACONDA_PATH + ".cache/datasets"
os.environ["TRANSFORMERS_CACHE"]=ANACONDA_PATH + ".cache/models"

from transformers import AutoProcessor
from vllm import LLM, SamplingParams
from qwen_vl_utils import process_vision_info
import torch

# Initialize the processor
#processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-72B-Instruct")
processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-72B-Instruct")

    
# Initialize the LLM
# from vllm.config import PoolerConfig
#llm = LLM(model="Qwen/Qwen2.5-VL-72B-Instruct", tensor_parallel_size=4, dtype=torch.bfloat16) # , task="generate", override_pooler_config=PoolerConfig(pooling_type="ALL"))
llm = LLM(model="Qwen/Qwen2.5-VL-72B-Instruct", tensor_parallel_size=4, dtype=torch.bfloat16) # , task="generate", override_pooler_config=PoolerConfig(pooling_type="ALL"))

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

def main():

    qwen = QWEN_VLM(llm, processor)
    
    folder_path = "results_qwen_72B"
    #folder_path = "results_qwen_72B_creditcards"
    files = glob.glob(folder_path + "/*")
    
    qwen.sampling_params = SamplingParams(
                temperature=0.1,
                top_p=0.001,
                repetition_penalty=1.05,
                max_tokens=5000,
                stop_token_ids=[],
                # logprobs=20,
            )     
    
    files.sort()
    
    meta_categories = ["bank statement", "letter with address", "credit or debit card", "bills or receipt", "preganancy test", "pregnancy test box", "mortage or investment report", "doctor prescription", "empty pill bottle", "condom with plastic bag", "tattoo sleeve", "transcript", "business card", "condom box", "local newspaper", "medical record document", "email", "phone", "id card"]
    
    if not os.path.exists(f"results_qwen_72B_img_categories/"):
        os.makedirs(f"results_qwen_72B_img_categories/")

            
    if not os.path.exists(f"results_qwen_72B_img_categories/all_meta_categories.json"):

        priv_images_categories = {}
        
        for ii in tqdm(range(0, len(files))):
            filepath = files[ii]
            with open(filepath, 'r') as file:
                data = json.load(file)
            output, _ = qwen.call_vlm(data['image_path'], f"From this list of categories: {' ,'.join(meta_categories)}, which one is related to this image, Only output the category")
            priv_images_categories[data['image_path']] = output
        
        filename = "all_meta_categories.json"
        with open(f"results_qwen_72B_img_categories/{filename}", "w") as fp:
            json.dump(priv_images_categories, fp, indent=4)
    

    if not os.path.exists(f"results_qwen_72B_img_categories/all_meta_categories_locations.json"):

        with open('results_qwen_72B_img_categories/all_meta_categories.json', 'r') as file:
            data = json.load(file)

        for ii, (k,v) in tqdm(enumerate(data.items())):
            output, _ = qwen.call_vlm(k, f"Locate the {v}. Output json with bboxes.")
            bbox = extract_bbox_removing_incomplete(output)
            data[k] = {'label': v, 'anns': bbox}
        
        filename = "all_meta_categories_locations.json"
        with open(f"results_qwen_72B_img_categories/{filename}", "w") as fp:
            json.dump(data, fp, indent=4)
            
    
    if not os.path.exists(f"results_qwen_72B_img_categories/all_meta_categories_locations_v2.json"):
        
        with open('results_qwen_72B_img_categories/all_meta_categories_locations.json', 'r') as file:
            data = json.load(file)
        
        for ii, (k, vals) in tqdm(enumerate(data.items())):
            short_description, _ = qwen.call_vlm(k, f"describe this image with three short sentences, include the private content in detail")
            long_description, _ = qwen.call_vlm(k, f"describe this image, including the private content in detail")
        
            data[k] = {'label': vals['label'], 'anns': vals['anns'], 'short_description': short_description, 'long_description': long_description}
        
        filename = "all_meta_categories_locations_v2.json"
        with open(f"results_qwen_72B_img_categories/{filename}", "w") as fp:
            json.dump(data, fp, indent=4)

    
    if not os.path.exists(f"results_qwen_72B_img_categories/all_meta_categories_locations_v3.json"):

        with open('results_qwen_72B_img_categories/all_meta_categories_locations_v2.json', 'r') as file:
            data = json.load(file)

        segmentation_agent = SEGMENTATION_AGENT()

        for ii, (file_path, info) in tqdm(enumerate(data.items())):
            try:
                cropped_image = crop_image_bbox(file_path, info['anns'][0]['bbox_2d'])
                segment_mask = segmentation_agent.segment_document_prompt(cropped_image,f"the {info['label']}", False)
                full_image_mask = get_full_object_mask(file_path, segment_mask, info['anns'][0]['bbox_2d'])
                masked_full_image = full_image_zero_outside_mask(file_path, ~full_image_mask)
        
                short_description, _ = qwen.call_vlm(masked_full_image, f"describe this image with three short sentences, include the private content in detail")
                long_description, _ = qwen.call_vlm(masked_full_image, f"describe this image, including the private content in detail")
        
                full_mask_vis = (full_image_mask * 1).astype(np.uint8)
                info['full_image_mask'] = image_to_base64_str(full_mask_vis)
                info['full_image_mask_short_description'] = short_description
                info['full_image_mask_long_description'] = long_description
                data[file_path] = info
            except:
                info['full_image_mask'] = 'None'
                info['full_image_mask_short_description'] = 'None'
                info['full_image_mask_long_description'] = 'None'
                data[file_path] = info
        
        filename = "all_meta_categories_locations_v3.json"
        with open(f"results_qwen_72B_img_categories/{filename}", "w") as fp:
            json.dump(data, fp, indent=4)


    if not os.path.exists(f"results_qwen_72B_img_categories/all_meta_categories_locations_v4.json"):

        with open('results_qwen_72B_img_categories/all_meta_categories_locations_v3.json', 'r') as file:
            data = json.load(file)

        with open('/gpfs/projects/CascanteBonillaGroup/paola/Qwen2.5-VL/results_qwen_72B_img_categories/3d_objects.pkl', 'rb') as file:
            all_3d_objects = pickle.load(file)

        
        for img_path, anns in tqdm(data.items()):
            if anns['full_image_mask'] != 'None':
                full_mask_vis = base64_str_to_image2(anns['full_image_mask'])
                full_mask_vis[full_mask_vis==1]=255
                full_mask_vis = Image.fromarray(full_mask_vis).convert("L")
                full_mask_vis = np.array(full_mask_vis)
        
                tryouts = 0
                found = False
                random_object_path = random.choice(list(all_3d_objects.keys()))
                obj = all_3d_objects[random_object_path]['rendered_image']
                while not found:
                    try:
                        pos_x, pos_y = random_position_not_on_mask(full_mask_vis, obj.size, max_attempts=1000)
                        found = True
                    except:
                        random_object_path = random.choice(list(all_3d_objects.keys()))
                        obj = all_3d_objects[random_object_path]['rendered_image']
                    tryouts += 1
                    if tryouts == 40:
                        pos_x, pos_y = random_position_not_on_mask(~full_mask_vis, obj.size, max_attempts=1000)
                        break            
                
                random_position = [pos_x, pos_y] # random.choice(valid_positions)
                
                anns['object_path'] = random_object_path
                anns['object_position'] = random_position
                anns['object_outside_mask'] = found
        
                data[img_path] = anns

        filename = "all_meta_categories_locations_v4.json"
        with open(f"results_qwen_72B_img_categories/{filename}", "w") as fp:
            json.dump(data, fp, indent=4)
                
if __name__ == '__main__':
    main()
