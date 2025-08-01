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
import math
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
from config import cfg


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
    parser.add_argument("--query", default='paper document')
    
    args2 = parser.parse_args([])
    args2.path  = cfg.data_path 
    #args2.query = "paper document"
    args2.query = "credit card"
    
    folder_path = args2.path
    files = glob.glob(folder_path + "/*")

    if args2.query ==  'paper document':
        if not os.path.exists(f"results_qwen_72B/"):
            os.makedirs(f"results_qwen_72B/")
            os.makedirs(f"results_qwen_72B_augmented/")
    else: 
        if not os.path.exists(f"results_qwen_72B_creditcards/"):
            os.makedirs(f"results_qwen_72B_creditcards/")
            os.makedirs(f"results_qwen_72B_creditcards_augmented/")
      
    
    for ii in tqdm(range(len(files))):
    
        doc_img_info = {}
        image_path = files[ii]
        doc_img_info["image_path"] = image_path

        if args2.query == 'paper document':
            response, pil_image = qwen.call_vlm(image_path, "Locate paper document in the image, and output in JSON format.", temperature = 1, top_p=0.8)
        else:
            response, pil_image = qwen.call_vlm(image_path, "Is there a credit card in the image? Answer yes or no.", temperature = 1, top_p=0.8)
            if not "yes" in response.lower():
                continue
            response, pil_image = qwen.call_vlm(image_path, "Locate credit card in the image, and output in JSON format.", temperature = 1, top_p=0.8)
            
    
    
        bbox_orig = qwen.extract_bbox(response)
        doc_img_info["doc_bbox"] = bbox_orig
        cropped_image = qwen.crop_image(image_path, bbox_orig)
        pred_mask = segmentation_agent.segment_document(cropped_image)
    
        masked_image = set_zero_outside_mask(pil_to_opencv(cropped_image), pred_mask)
        masked_image_copy = masked_image.copy()
        doc_img_info["masked_image"] = image_to_base64_str(masked_image_copy)
        masked_image_pil = opencv_to_pil(masked_image)
    
        top_plausible_rotations = orientation_agent.detect_rotation(masked_image_pil, np.arange(0, 360+30, 30))
        rotate_angle = top_plausible_rotations[0]
        doc_img_info["initial_rotate_angle"] = rotate_angle
        # print (rotate_angle)
    
        # rotate_angle = 30
        cropped_image_tmp = masked_image_pil.copy()
        rotated_image_v1 = cropped_image_tmp.rotate(rotate_angle, resample=Image.BICUBIC, expand=True, fillcolor=(255,255,255))
        # rotated_image_v1 = masked_image_pil.copy()
        rotated_image_v1.save("rotated_image.jpg")
        points, strs, elapse = pladdleOCR()
    
        angles = []
        for pol in points[:10]:
            angles.append(polygon_orientation(pol))
    
        mean_angle = np.mean(angles)
        # print (mean_angle)

    
        total_angle = rotate_angle
        detailed_data = []
        if angles: ## if the ocr detects something to correct the angle, if not, leave as it is an get the vlm outputs

            total_angle += mean_angle
            rotated_image_v1 = cropped_image_tmp.rotate(total_angle, resample=Image.BICUBIC, expand=True, fillcolor=(255,255,255))
            rotated_image_v1.save("rotated_image.jpg")
            points, strs, elapse = pladdleOCR()

        bboxes_text_info = qwen.get_finegrained_text(rotated_image_v1)
        data = extract_bbox_removing_incomplete(bboxes_text_info)
        #print(pil_to_opencv(rotated_image_v1))
        detailed_data.append({"rotate_angle": total_angle, "data_vlm": data, "data_ocr": [points, strs], "rotated_image": image_to_base64_str(pil_to_opencv(rotated_image_v1))})
        doc_img_info["detailed_data"] = detailed_data
    
    
        filename = os.path.basename(image_path).replace(".jpeg", ".json")
        doc_img_info_converted = convert_np(doc_img_info)
    
        if args2.query == 'paper document':
            with open(f"results_qwen_72B/{filename}", "w") as fp:
                json.dump(doc_img_info_converted, fp, indent=4)

            #with open(f"results_qwen_72B/{filename}", 'r') as file:
            #    data = json.load(file)
        else:
            with open(f"results_qwen_72B_creditcards/{filename}", "w") as fp:
                json.dump(doc_img_info_converted, fp, indent=4)

            #with open(f"results_qwen_72B_creditcards/{filename}", 'r') as file:
            #    data = json.load(file)

        data = doc_img_info_converted
        original_detailed_data = {'data_vlm':[[],[]], 'data_ocr':[[],[]]}

        #print(data)
    
        if data['detailed_data'][0]['data_vlm'] is not None:#data['doc_bbox'] is not None and data['detailed_data'][0]['data_vlm'] is not None:
      
            img = Image.open(image_path)
            rotated_image_cv  = base64_str_to_image2(data['detailed_data'][0]['rotated_image'])
            rotated_image_pil = opencv_to_pil(rotated_image_cv)
            rot_im = rotated_image_pil.copy()
            img_crop = img.crop(data['doc_bbox'])
           
            if data['doc_bbox'] is None:
                crop_x = 0
                crop_y = 0
            else:
                crop_x = data['doc_bbox'][0]
                crop_y = data['doc_bbox'][1]
    
            rotate_angle = -data['detailed_data'][0]['rotate_angle'] # -32  # example angle in degrees
            
            polygons = []
            texts = []
            im  = img.copy()
            for bbox_rotated_tmp in data['detailed_data'][0]['data_vlm']:
                if 'bbox_2d' in bbox_rotated_tmp and 'text_content' in bbox_rotated_tmp and isinstance(bbox_rotated_tmp['bbox_2d'], list):
                    bbox_rotated = bbox_rotated_tmp['bbox_2d']
                    text = bbox_rotated_tmp['text_content']
            
                    if len(bbox_rotated) != 4:
                        continue
    
                    poly_orig = rotated_bbox_polygon(bbox_rotated, rotate_angle, img_crop.size, rot_im.size)
            
                    final_poly = []
                    for (x,y) in poly_orig:
                        final_poly.append((crop_x+x, crop_y+y))
            
                    draw_orig = ImageDraw.Draw(im)
                    draw_orig.polygon(final_poly, outline="red", width=2)
            
                    polygons.append(np.array(final_poly).tolist())
                    texts.append(text)
            
            
            original_detailed_data['data_vlm'][0].extend(polygons)
            original_detailed_data['data_vlm'][1].extend(texts)
            
            im = img.copy()
            draw_orig = ImageDraw.Draw(im)
            
            polygons = []
            texts = []
            texts = data['detailed_data'][0]['data_ocr'][1]
            
            for ocr_polygon in data['detailed_data'][0]['data_ocr'][0]:
                poly_orig = rotated_polygon_points(ocr_polygon, rotate_angle, img_crop.size, rot_im.size)
            
                final_poly = []
                for (x,y) in poly_orig:
                    final_poly.append((crop_x+x, crop_y+y))
            
                draw_orig.polygon(final_poly, outline="green", width=2)
                polygons.append(np.array(final_poly).tolist())
            
            original_detailed_data['data_ocr'][0].extend(polygons)
            original_detailed_data['data_ocr'][1].extend(texts)
            

        data['original_detailed_data'] = [original_detailed_data]
        if args2.query == 'paper document':
            with open(f"results_qwen_72B_augmented/{filename}", "w") as fp:
                json.dump(data, fp, indent=4)
        else:
            with open(f"results_qwen_72B_creditcards_augmented/{filename}", "w") as fp:
                json.dump(data, fp, indent=4)


if __name__ == "__main__":
    main()
