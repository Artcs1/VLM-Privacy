import os
import sys

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3"

ANACONDA_PATH = "/gpfs/projects/CascanteBonillaGroup/jmurrugarral/anaconda3/"

os.environ["PIP_CACHE_DIR"]= ANACONDA_PATH +".cache"
os.environ["HF_HOME"]=ANACONDA_PATH + ".cache"
os.environ["HF_DATASETS_CACHE"]=ANACONDA_PATH + ".cache/datasets"
os.environ["TRANSFORMERS_CACHE"]=ANACONDA_PATH + ".cache/models"

#__dir__ = "/gpfs/projects/CascanteBonillaGroup/paola/PaddleOCR"
#sys.path.append(__dir__)
#sys.path.insert(0, os.path.abspath(os.path.join(__dir__, "../..")))
#sys.path.append("/gpfs/projects/CascanteBonillaGroup/paola/EVF-SAM/")

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




def main():

    additional_colors = [colorname for (colorname, colorcode) in ImageColor.colormap.items()]
    
    # default: Load the model on the available device(s)
    model_7b = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-VL-7B-Instruct",
        torch_dtype=torch.bfloat16,
        # attn_implementation="flash_attention_2",
        device_map="cuda:2",
    )
    
    
    # default processer
    processor_7b = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
    model_7b.eval()
    
    qwen = QWEN_VLM(llm, processor)
    orientation_agent  = ORIENTATION_AGENT(model_7b, processor_7b)
    segmentation_agent = SEGMENTATION_AGENT()
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--path")
    parser.add_argument('--do_rotation', action='store_true', help="Set this flag to True")
    
    args2 = parser.parse_args([])
    # args2.path = "/gpfs/projects/CascanteBonillaGroup/datasets/BIV-Priv_Image/support_images"
    args2.path = "/gpfs/projects/CascanteBonillaGroup/datasets/BIV-Priv_Image/query_images" # DATASET PATH
    args2.do_rotation = True
    
    folder_path = args2.path
    files = glob.glob(folder_path + "/*")

    length = len(files)
    n_trials = 100
    n_files  = [random.randint(1, length) for _ in range(n_trials)]

   
     
    
    start_time = time.time()
    all_times = []
    #for ii in range(length):
    for ii in n_files:
        #print(ii)

        ex_stime = time.time()

        image_path = files[ii]
        response, pil_image = qwen.call_vlm(image_path, "Locate paper document in the image, and output in JSON format.", temperature = 1, top_p=0.8)
    
        bbox_orig = qwen.extract_bbox(response)
        cropped_image = qwen.crop_image(image_path, bbox_orig)
        pred_mask = segmentation_agent.segment_document(cropped_image)
    
        masked_image = set_zero_outside_mask(pil_to_opencv(cropped_image), pred_mask)
        masked_image_copy = masked_image.copy()
        masked_image_pil = opencv_to_pil(masked_image)
    
        top_plausible_rotations = orientation_agent.detect_rotation(masked_image_pil, np.arange(0, 360+30, 30))
        rotate_angle = top_plausible_rotations[0]
        # print (rotate_angle)
    
        # rotate_angle = 30
        cropped_image_tmp = masked_image_pil.copy()
        rotated_image_v1 = cropped_image_tmp.rotate(rotate_angle, resample=Image.BICUBIC, expand=True, fillcolor=(255,255,255))

        rotated_image_v1.save("rotated_image.jpg")
        points, strs, elapse = pladdleOCR()
    
        angles = []
        for pol in points[:10]:
            angles.append(polygon_orientation(pol))
    
        mean_angle = np.mean(angles)
        # print (mean_angle)
    
        total_angle = rotate_angle + mean_angle 

        detailed_data = []
        if angles: ## if the ocr detects something to correct the angle, if not, leave as it is an get the vlm outputs

            cropped_image_tmp    = masked_image_pil.copy()
            rotated_image_v1_tmp = cropped_image_tmp.rotate(total_angle, resample=Image.BICUBIC, expand=True, fillcolor=(255,255,255))
            rotated_image_v1     = rotated_image_v1_tmp
    
            bboxes_text_info     = qwen.get_finegrained_text(rotated_image_v1)
            
            data = extract_bbox_removing_incomplete(bboxes_text_info)
            rotated_image_v1.save("rotated_image.jpg")
            points, strs, elapse = pladdleOCR()
    
        else:

            bboxes_text_info = qwen.get_finegrained_text(rotated_image_v1)
            data = extract_bbox_removing_incomplete(bboxes_text_info)

    
        meta_categories = ["bank statement", "letter with address", "credit or debit card", "bills or receipt", "preganancy test", "pregnancy test box","mortage or investment report", "doctor prescription", "empty pill bottle", "condom with plastic bag", "tattoo sleeve", "transcript","business card", "condom box", "local newspaper", "medical record document", "email", "phone", "id card",]

        meta_category, _ = qwen.call_vlm(pil_image, f"From this list of categories: {' ,'.join(meta_categories)}, which one is related to this image> Only output the category")
        #print(meta_category)

        if data is not None:

            texts = [d['text_content']for d in data]
            #print(texts)

            high_risk = []
            for text in texts:
                proposed_labels_3 = qwen.guided_labeling_per_match_per_metacategory(rotated_image_v1, text, meta_category)
                high_risk.append(proposed_labels_3)
                #print(proposed_labels_3)

            #print(high_risk)


        ex_etime = time.time()
        all_times.append(ex_etime-ex_stime)

       
    all_times = np.array(all_times)

    end_time = time.time()

    total_time = end_time - start_time
    avg_fps = n_trials / total_time
    print(f"Average FPS: {avg_fps:.2f}")

    time_per_image = total_time / n_trials
    print(f"Average time per image: {time_per_image:.4f} seconds")

    print(f"Min time: {np.min(all_times):.2f} seconds")
    print(f"Max time: {np.max(all_times):.2f} seconds")
    print(f"std time: {np.std(all_times):.2f} seconds")

    np.save('my_array.npy', all_times)

if __name__ == "__main__":
    main()
