import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3"


os.environ["PIP_CACHE_DIR"]="/gpfs/projects/CascanteBonillaGroup/jmurrugarral/anaconda3/.cache"
os.environ["HF_HOME"]="/gpfs/projects/CascanteBonillaGroup/jmurrugarral/anaconda3/.cache"
os.environ["HF_DATASETS_CACHE"]="/gpfs/projects/CascanteBonillaGroup/jmurrugarral/anaconda3/.cache/datasets"
os.environ["TRANSFORMERS_CACHE"]="/gpfs/projects/CascanteBonillaGroup/jmurrugarral/anaconda3/.cache/models"



from transformers import AutoProcessor
from vllm import LLM, SamplingParams
from qwen_vl_utils import process_vision_info
import torch

# Initialize the processor
processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-72B-Instruct")



# Initialize the LLM
# from vllm.config import PoolerConfig
llm = LLM(model="Qwen/Qwen2.5-VL-72B-Instruct", tensor_parallel_size=4, dtype=torch.bfloat16) # , task="generate", override_pooler_config=PoolerConfig(pooling_type="ALL"))

# llm = LLM(model="Qwen/Qwen2.5-VL-72B-Instruct", torch_dtype=torch.bfloat16,
#         attn_implementation="flash_attention_2",
#         device_map="auto",
#     )

import re
import os
import argparse
import glob
import time
import json
from IPython.display import display   

import torch
import json
import random
import io
import ast
from PIL import Image, ImageDraw, ImageFont
from PIL import ImageColor


from shapely.validation import explain_validity
from shapely.geometry import Polygon
import cv2
import numpy as np
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
from collections import Counter


import io
import pandas as pd
import base64

from agents.qwen import QWEN_VLM
from agents.orientation import ORIENTATION_AGENT
from agents.segmentation import SEGMENTATION_AGENT
from agents.ocr import OCR_AGENT, pladdleOCR
from utils import *



from tqdm import tqdm

qwen = QWEN_VLM(llm, processor)

with open("results_qwen_72B_img_categories/all_meta_categories.json", 'r') as file:
    all_meta_categories = json.load(file)


paths = ['results_qwen_72B_creditcards', 'results_qwen_72B']
for folder_path in paths:

    files = glob.glob(folder_path + "/*")
    files.sort()
    
    all_images_finegrained_labeled = {}
    problematic_files = []
    
    for ii, ff in tqdm(enumerate(files)):
    
        with open(ff, 'r') as file:
            vlm_output_data = json.load(file)
    
        try:
            dict_finegrained_labeled = {}
            for vlm_finegrained in vlm_output_data['detailed_data'][0]['data_vlm']:
                for kk in vlm_finegrained.keys():
                    if kk != 'bbox_2d':
                        break
                
                filename = os.path.basename(ff)
                filename_metacategory = "/gpfs/projects/CascanteBonillaGroup/datasets/BIV-Priv_Image/all_images/"+filename.replace(".json", ".jpeg")
                res = qwen.guided_labeling_per_match_per_metacategory(vlm_output_data['image_path'], vlm_finegrained[kk], all_meta_categories[filename_metacategory])
                dict_finegrained_labeled[vlm_finegrained[kk]] = res
                # res = qwen.guided_labeling_per_match(vlm_output_data['image_path'], vlm_finegrained[kk])
                # dict_finegrained_labeled[vlm_finegrained[kk]] = res
                
            all_images_finegrained_labeled[vlm_output_data['image_path']] = {"image_category": all_meta_categories[filename_metacategory], "finegrained_labels": dict_finegrained_labeled}
            
            if 'creditcards' in folder_path:
                filename = f"creditcards_fine_grained_labels_per_metacategory.json"
                with open(f"results_qwen_72B_img_categories/{filename}", "w") as fp:
                    json.dump(all_images_finegrained_labeled, fp, indent=4, cls=NumpyEncoder)
            else:
                filename = f"documents_fine_grained_labels_per_metacategory.json"
                with open(f"results_qwen_72B_img_categories/{filename}", "w") as fp:
                    json.dump(all_images_finegrained_labeled, fp, indent=4, cls=NumpyEncoder)
            
        
            # break
        except:
            print(ii)
            problematic_files.append(ff)
            # break
    
    
