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


from utils import pil_to_opencv


class SEGMENTATION_AGENT():

    def __init__(self):

        self.VERSION = "YxZhang/evf-sam"
        self.VIS_SAVE_PATH = "./vis"
        self.PRECISION = "fp16" # bf32, fp16, fp32
        self.IMAGE_SIZE = 224
        self.MODEL_MAX_LEN = 512
        self.LOCAL_RANK = 0
        self.LOAD_IN_4BIT = False
        self.LOAD_IN_8BIT = False
        self.MODEL_TYPE = "ori" # ori, effi, sam2
        tokenizer_evf_sam, model_evf_sam = self.init_models()
        self.tokenizer_evf_sam = tokenizer_evf_sam
        self.model_evf_sam = model_evf_sam

    def init_models(self):
        tokenizer = AutoTokenizer.from_pretrained(
            self.VERSION,
            padding_side="right",
            use_fast=False,
        )
    
        torch_dtype = torch.float32
        if self.PRECISION == "bf16":
            torch_dtype = torch.bfloat16
        elif self.PRECISION == "fp16":
            torch_dtype = torch.half
    
        kwargs = {"torch_dtype": torch_dtype}
        if self.LOAD_IN_4BIT:
            kwargs.update(
                {
                    "torch_dtype": torch.half,
                    "quantization_config": BitsAndBytesConfig(
                        llm_int8_skip_modules=["visual_model"],
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.float16,
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_quant_type="nf4",
                    ),
                }
            )
        elif self.LOAD_IN_8BIT:
            kwargs.update(
                {
                    "torch_dtype": torch.half,
                    "quantization_config": BitsAndBytesConfig(
                        llm_int8_skip_modules=["visual_model"],
                        load_in_8bit=True,
                    ),
                }
            )
    
        if self.MODEL_TYPE=="ori":
            from model.evf_sam import EvfSamModel
            model = EvfSamModel.from_pretrained(
                self.VERSION, low_cpu_mem_usage=True, **kwargs
            )
        elif self.MODEL_TYPE=="effi":
            from model.evf_effisam import EvfEffiSamModel
            model = EvfEffiSamModel.from_pretrained(
                self.VERSION, low_cpu_mem_usage=True, **kwargs
            )
        elif self.MODEL_TYPE=="sam2":
            from model.evf_sam2 import EvfSam2Model
            model = EvfSam2Model.from_pretrained(
                self.VERSION, low_cpu_mem_usage=True, **kwargs
            )
    
        if (not self.LOAD_IN_4BIT) and (not self.LOAD_IN_8BIT):
            model = model.cuda()
        model.eval()
    
        return tokenizer, model

    def sam_preprocess(
        self,
        x: np.ndarray,
        pixel_mean=torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1),
        pixel_std=torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1),
        img_size=1024,
        model_type="ori") -> torch.Tensor:
        '''
        preprocess of Segment Anything Model, including scaling, normalization and padding.
        preprocess differs between SAM and Effi-SAM, where Effi-SAM use no padding.
        input: ndarray
        output: torch.Tensor
        '''
        assert img_size==1024, \
            "both SAM and Effi-SAM receive images of size 1024^2, don't change this setting unless you're sure that your employed model works well with another size."
        x = ResizeLongestSide(img_size).apply_image(x)
        resize_shape = x.shape[:2]
        x = torch.from_numpy(x).permute(2,0,1).contiguous()
    
        # Normalize colors
        x = (x - pixel_mean) / pixel_std
        if model_type=="effi" or model_type=="sam2":
            x = F.interpolate(x.unsqueeze(0), (img_size, img_size), mode="bilinear").squeeze(0)
        else:
            # Pad
            h, w = x.shape[-2:]
            padh = img_size - h
            padw = img_size - w
            x = F.pad(x, (0, padw, 0, padh))
        return x, resize_shape
    
    def beit3_preprocess(self, x: np.ndarray, img_size=224) -> torch.Tensor:
        '''
        preprocess for BEIT-3 model.
        input: ndarray
        output: torch.Tensor
        '''
        beit_preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((img_size, img_size), interpolation=InterpolationMode.BICUBIC),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])
        return beit_preprocess(x)

    def segment_document(self,cropped_image):

        PROMPT = "the document"
        # Preprocess
        cropped_image_tmp = cropped_image.copy()
        image_np = pil_to_opencv(cropped_image_tmp)
        image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        original_size_list = [image_np.shape[:2]]

        image_beit = self.beit3_preprocess(image_np, self.IMAGE_SIZE).to(dtype=self.model_evf_sam.dtype, device=self.model_evf_sam.device)
        image_sam, resize_shape = self.sam_preprocess(image_np, model_type=self.MODEL_TYPE)
        image_sam = image_sam.to(dtype=self.model_evf_sam.dtype, device=self.model_evf_sam.device)
        input_ids = self.tokenizer_evf_sam(PROMPT, return_tensors="pt")["input_ids"].to(device=self.model_evf_sam.device)

        # inference
        pred_mask = self.model_evf_sam.inference(
            image_sam.unsqueeze(0),
            image_beit.unsqueeze(0),
            input_ids,
            resize_list=[resize_shape],
            original_size_list=original_size_list,
        )

        pred_mask = pred_mask.detach().cpu().numpy()[0]
        pred_mask = pred_mask > 0

        # # Show visualization
        # save_img = image_np.copy()
        # save_img[pred_mask] = (
        #     image_np * 0.5
        #     + pred_mask[:, :, None].astype(np.uint8) * np.array([50, 120, 220]) * 0.5
        # )[pred_mask]
        # save_img = cv2.cvtColor(save_img, cv2.COLOR_RGB2BGR)

        # plt.imshow(cv2.cvtColor(save_img, cv2.COLOR_BGR2RGB))
        # plt.axis("off")
        # plt.show()
        return pred_mask
