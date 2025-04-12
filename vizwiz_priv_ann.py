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

    # Initialize the processor
    #processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-72B-Instruct")
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
    
    # Initialize the LLM
    # from vllm.config import PoolerConfig
    #llm = LLM(model="Qwen/Qwen2.5-VL-72B-Instruct", tensor_parallel_size=4, dtype=torch.bfloat16) # , task="generate", override_pooler_config=PoolerConfig(pooling_type="ALL"))
    llm = LLM(model="Qwen/Qwen2.5-VL-7B-Instruct", tensor_parallel_size=2, dtype=torch.bfloat16) # , task="generate", override_pooler_config=PoolerConfig(pooling_type="ALL"))
    
    # llm = LLM(model="Qwen/Qwen2.5-VL-72B-Instruct", torch_dtype=torch.bfloat16,
    #         attn_implementation="flash_attention_2",
    #         device_map="auto",
    #     )
    
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
    args2.path = "/gpfs/projects/CascanteBonillaGroup/datasets/BIV-Priv_Image/query_images"
    args2.do_rotation = True
    
    folder_path = args2.path
    files = glob.glob(folder_path + "/*")
    
    if not os.path.exists(f"results_qwen_72B/"):
        os.makedirs(f"results_qwen_72B/")
       
    
    for ii in range (5, len(files)):
    
        print(ii)
        doc_img_info = {}
        image_path = files[ii]
        doc_img_info["image_path"] = image_path
    
        response, pil_image = qwen.call_vlm(image_path, "Locate paper document in the image, and output in JSON format.", temperature = 1, top_p=0.8)
    
        bbox_orig = qwen.extract_bbox(response)
        doc_img_info["doc_bbox"] = bbox_orig
        # h_margin = 0
        # v_margin = 0
        # bbox = [bbox_orig[0]-h_margin, bbox_orig[1]-v_margin, bbox_orig[2]+h_margin, bbox_orig[3]+v_margin]
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
            new_angle = mean_angle
            # for new_angle in [mean_angle]: #, 180]:
            rotated_image_v1_tmp = rotated_image_v1.copy()
            rotated_image_v1_tmp = rotated_image_v1.rotate(new_angle, resample=Image.BICUBIC, expand=True, fillcolor=(255,255,255))
            total_angle += new_angle
            rotated_image_v1 = rotated_image_v1_tmp
            # doc_img_info["rotate_angle"] = mean_angle # rotate_angle + mean_angle
    
            # print (f"Angle: {rotate_angle + mean_angle}\n---")
            bboxes_text_info = qwen.get_finegrained_text(rotated_image_v1)
            data = extract_bbox_removing_incomplete(bboxes_text_info)
            # plot_text_inside_image(rotated_image_v1, data)
    
            rotated_image_v1.save("rotated_image.jpg")
            points, strs, elapse = pladdleOCR()
    
            detailed_data.append({"rotate_angle": total_angle, "data_vlm": data, "data_ocr": [points, strs], "rotated_image": image_to_base64_str(pil_to_opencv(rotated_image_v1))})
    
            # src_im = utility.draw_e2e_res(points, strs, "rotated_image.jpg")
            # plt.figure(figsize=(20, 20))
            # plt.imshow(src_im)
            # plt.show()
    
            doc_img_info["detailed_data"] = detailed_data
        else:
            new_angle = 0
            total_angle += new_angle
    
            bboxes_text_info = qwen.get_finegrained_text(rotated_image_v1)
            data = extract_bbox_removing_incomplete(bboxes_text_info)
    
            detailed_data.append({"rotate_angle": total_angle, "data_vlm": data, "data_ocr": [points, strs], "rotated_image": image_to_base64_str(pil_to_opencv(rotated_image_v1))})
    
            doc_img_info["detailed_data"] = detailed_data
    
    
        filename = os.path.basename(image_path).replace(".jpeg", ".json")
        doc_img_info_converted = convert_np(doc_img_info)
        with open(f"results_qwen_72B/{filename}", "w") as fp:
            json.dump(doc_img_info_converted, fp, indent=4)

if __name__ == "__main__":
    main()
