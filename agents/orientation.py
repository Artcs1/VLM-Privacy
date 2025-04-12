
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
from ppocr.utils.utility import get_image_file_list, check_and_read
from ppocr.data import create_operators, transform
from ppocr.postprocess import build_post_process
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

class ORIENTATION_AGENT():

    def __init__(self, model_7b, processor_7b):
        self.model_7b = model_7b
        self.processor_7b = processor_7b

    def detect_rotation(self,cropped_image, plausible_rotations):
    
        yes_probs = []
    
        for rotation_ii in plausible_rotations:
            cropped_image_tmp = cropped_image.copy()
            rotated_image = cropped_image_tmp.rotate(rotation_ii, resample=Image.BICUBIC, expand=True, fillcolor=(255,255,255))
    
            # _, rotated_image = rotate_image(pil_to_opencv(cropped_image_tmp), rotation_ii, 'rotated.jpg')
    
            #plt.imshow(rotated_image)
            #plt.show()
            
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": rotated_image,
                            "min_pixels": 224 * 224,
                            "max_pixels": 1280 * 28 * 28,
                        },
                    # {"type": "text", "text": "Is this document properly aligned? Answer yes or no."},
                    {"type": "text", "text": "Is the text in this document readable (top down, left to right)? Answer yes or no."},
                    # {"type": "text", "text": "Is the document properly aligned? (text reads from top to down and left to right)? Answer yes or no."},
                ],
                }
            ]
    
            # Preparation for inference
            text = self.processor_7b.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.processor_7b(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to(self.model_7b.device)
    
            # Inference: Generation of the output
            with torch.no_grad():
                outputs = self.model_7b.generate(**inputs, max_new_tokens=128,
                                        return_dict_in_generate=True,
                                        use_cache=False,
                                        # return_dict=True,
                                        output_scores=True,
                                        do_sample=False)
                generated_ids = outputs["sequences"]
                generated_ids_trimmed = [
                    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                output_text = self.processor_7b.batch_decode(
                    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )
    
                inputs_for_prob = self.processor_7b(text=["Yes"],
                            images=image_inputs,
                            videos=video_inputs,
                            padding=True,
                            return_tensors="pt")
                yes_id = inputs_for_prob['input_ids'][0][0].item()
        
                inputs_for_prob = self.processor_7b(text=["No"],
                                images=image_inputs,
                                videos=video_inputs,
                                padding=True,
                                return_tensors="pt")
                no_id = inputs_for_prob['input_ids'][0][0].item()
    
                logits = outputs["scores"][0][0]
                probs = (torch.nn.functional.softmax(torch.tensor([logits[yes_id], logits[no_id]]), dim=0).detach().cpu().numpy())
                # print(f"Output text: {output_text} :: {rotation_ii} - Yes prob: {probs[0]}, No prob: {probs[1]}")
                yes_probs.append(probs[0])
    
        # get two max scores:
        max_idxs = sorted(range(len(yes_probs)), key=lambda i: yes_probs[i], reverse=True)[:5]
    
        return plausible_rotations[max_idxs]

