import os
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor, AutoModel 
from qwen_vl_utils import process_vision_info

import re
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
import os
import sys


#import Levenshtein
from shapely.validation import explain_validity
from shapely.geometry import Polygon
import cv2
import numpy as np
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
from collections import Counter

import tools.infer.utility as utility
from ppocr.utils.logging import get_logger
from ppocr.utils.utility import get_image_file_list, check_and_read
from ppocr.data import create_operators, transform
from ppocr.postprocess import build_post_process


import io
import pandas as pd
import numpy as np

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"]="2"

additional_colors = [colorname for (colorname, colorcode) in ImageColor.colormap.items()]

def pil_to_opencv(img):
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

def opencv_to_pil(img):
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

# __dir__ = os.path.dirname(os.path.abspath(__file__))
__dir__ = os.path.dirname(os.path.abspath(os.getcwd()))
sys.path.append(__dir__)
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, "../..")))
#sys.path.insert(0, os.path.abspath(os.path.join(__dir__, "PaddleOCR")))


os.environ["FLAGS_allocator_strategy"] = "auto_growth"
logger = get_logger()




class TextE2E(object):
    def __init__(self, args):
        self.args = args
        self.e2e_algorithm = args.e2e_algorithm
        self.use_onnx = args.use_onnx
        pre_process_list = [
            {"E2EResizeForTest": {}},
            {
                "NormalizeImage": {
                    "std": [0.229, 0.224, 0.225],
                    "mean": [0.485, 0.456, 0.406],
                    "scale": "1./255.",
                    "order": "hwc",
                }
            },
            {"ToCHWImage": None},
            {"KeepKeys": {"keep_keys": ["image", "shape"]}},
        ]
        postprocess_params = {}
        if self.e2e_algorithm == "PGNet":
            pre_process_list[0] = {
                "E2EResizeForTest": {
                    "max_side_len": args.e2e_limit_side_len,
                    "valid_set": "totaltext",
                }
            }
            postprocess_params["name"] = "PGPostProcess"
            postprocess_params["score_thresh"] = args.e2e_pgnet_score_thresh
            postprocess_params["character_dict_path"] = args.e2e_char_dict_path
            postprocess_params["valid_set"] = args.e2e_pgnet_valid_set
            postprocess_params["mode"] = args.e2e_pgnet_mode
        else:
            logger.info("unknown e2e_algorithm:{}".format(self.e2e_algorithm))
            sys.exit(0)

        self.preprocess_op = create_operators(pre_process_list)
        self.postprocess_op = build_post_process(postprocess_params)
        (
            self.predictor,
            self.input_tensor,
            self.output_tensors,
            _,
        ) = utility.create_predictor(
            args, "e2e", logger
        )  # paddle.jit.load(args.det_model_dir)
        # self.predictor.eval()

    def clip_det_res(self, points, img_height, img_width):
        for pno in range(points.shape[0]):
            points[pno, 0] = int(min(max(points[pno, 0], 0), img_width - 1))
            points[pno, 1] = int(min(max(points[pno, 1], 0), img_height - 1))
        return points

    def filter_tag_det_res_only_clip(self, dt_boxes, image_shape):
        img_height, img_width = image_shape[0:2]
        dt_boxes_new = []
        for box in dt_boxes:
            box = self.clip_det_res(box, img_height, img_width)
            dt_boxes_new.append(box)
        dt_boxes = np.array(dt_boxes_new)
        return dt_boxes

    def __call__(self, img):
        ori_im = img.copy()
        data = {"image": img}
        data = transform(data, self.preprocess_op)
        img, shape_list = data
        if img is None:
            return None, 0
        img = np.expand_dims(img, axis=0)
        shape_list = np.expand_dims(shape_list, axis=0)
        img = img.copy()
        starttime = time.time()

        if self.use_onnx:
            input_dict = {}
            input_dict[self.input_tensor.name] = img
            outputs = self.predictor.run(self.output_tensors, input_dict)
            preds = {}
            preds["f_border"] = outputs[0]
            preds["f_char"] = outputs[1]
            preds["f_direction"] = outputs[2]
            preds["f_score"] = outputs[3]
        else:
            self.input_tensor.copy_from_cpu(img)
            self.predictor.run()
            outputs = []
            for output_tensor in self.output_tensors:
                output = output_tensor.copy_to_cpu()
                outputs.append(output)

            preds = {}
            if self.e2e_algorithm == "PGNet":
                preds["f_border"] = outputs[0]
                preds["f_char"] = outputs[1]
                preds["f_direction"] = outputs[2]
                preds["f_score"] = outputs[3]
            else:
                raise NotImplementedError
        post_result = self.postprocess_op(preds, shape_list)
        #print(post_result)
        points, strs = post_result["points"], post_result["texts"]#, post_result["confs"]
        dt_boxes = self.filter_tag_det_res_only_clip(points, ori_im.shape)
        elapse = time.time() - starttime
        return dt_boxes, strs, elapse


args = utility.parse_args()
print(args.e2e_algorithm)
args.e2e_algorithm="PGNet"
args.image_dir='../cropped_image.jpg'
args.e2e_model_dir="./inference/e2e_server_pgnetA_infer/"
args.use_gpu=False
args.e2e_pgnet_valid_set="totaltext"

text_detector = TextE2E(args)

draw_img_save = "./inference_results"
if not os.path.exists(draw_img_save):
    os.makedirs(draw_img_save)



class QWEN_VLM:

    def __init__(self):
        
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2.5-VL-7B-Instruct", 
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="auto")

        self.processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
        self.model.eval()        
 
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
    
    def guided_labeling(self, image):
    
        messages = [
        {"role": "system", "content": """You are a helpful assistant."""
        },
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
    
        
        # Preparation for inference
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.model.device)
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=500,
                                        return_dict_in_generate=True,)
                                        # use_cache=False,
                                        # return_dict=True,
                                        # output_scores=True, 
                                        # do_sample=False)
            generated_ids = outputs["sequences"]
            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
    
        return output_text[0]

   
    def plot_bounding_boxes(self, im, bounding_boxes, input_width, input_height, image_width, image_height, json=False):
        """
        Plots bounding boxes on an image with markers for each a name, using PIL, normalized coordinates, and different colors.
    
        Args:
            img_path: The path to the image file.
            bounding_boxes: A list of bounding boxes containing the name of the object
             and their positions in normalized [y1 x1 y2 x2] format.
        """
    
        # Load the image
        img = im
        width, height = img.size
        # Create a drawing object
        draw = ImageDraw.Draw(img)
    
        # Define a list of colors
        colors = ['red','green','blue','yellow','orange','pink','purple','brown','gray','beige','turquoise','cyan','magenta','lime','navy','maroon',
        'teal','olive','coral','lavender','violet','gold','silver',] + additional_colors
    
        if json == False:
    
            # Parsing out the markdown fencing
            bounding_boxes = self.parse_json(bounding_boxes)
    
            #font = ImageFont.truetype("NotoSansCJK-Regular.ttc", size=14)
    
            try:
              json_output = ast.literal_eval(bounding_boxes)
            except Exception as e:
              end_idx = bounding_boxes.rfind('"}') + len('"}')
              truncated_text = bounding_boxes[:end_idx] + "]"
              json_output = ast.literal_eval(truncated_text)
        else:
            #json_output = ast.literal_eval(bounding_boxes)
            json_output = bounding_boxes
    
        bboxes = np.zeros((int(len(json_output)),4))
        labels = []
    
        # Iterate over the bounding boxes
        for i, bounding_box in enumerate(json_output):
          # Select a color from the list
          color = colors[i % len(colors)]
    
          bboxes[i,:] = np.array([int(bounding_box["bbox_2d"][0]/input_width * image_width),
                                  int(bounding_box["bbox_2d"][1]/input_height * image_height),
                                  int(bounding_box["bbox_2d"][2]/input_width * image_width),
                                  int(bounding_box["bbox_2d"][3]/input_height * image_height)])
    
    
    
          # Convert normalized coordinates to absolute coordinates
          abs_y1 = int(bounding_box["bbox_2d"][1]/input_height * height)
          abs_x1 = int(bounding_box["bbox_2d"][0]/input_width * width)
          abs_y2 = int(bounding_box["bbox_2d"][3]/input_height * height)
          abs_x2 = int(bounding_box["bbox_2d"][2]/input_width * width)
    
          if abs_x1 > abs_x2:
            abs_x1, abs_x2 = abs_x2, abs_x1
            bboxes[i,0], bboxes[i,2] = bboxes[i,2], bboxes[i,0]
    
    
          if abs_y1 > abs_y2:
            abs_y1, abs_y2 = abs_y2, abs_y1
            bboxes[i,1], bboxes[i,3] = bboxes[i,3], bboxes[i,1]
    
          #bboxes[i,:] = np.array([abs_x1, abs_y1, abs_x2, abs_y2])
    
          # Draw the bounding box
          draw.rectangle(((abs_x1, abs_y1), (abs_x2, abs_y2)), outline=color, width=4)
    
          # Draw the text
          if "label" in bounding_box:
            draw.text((abs_x1 + 8, abs_y1 + 6), bounding_box["label"], fill=color)#, font=font)
    
          if "text_content" in bounding_box:
            labels.append(bounding_box["text_content"])
    
        # Display the image
        #img.show()
    
        return img, bboxes, labels
    
    # @title Parsing JSON output
    def parse_json(self, json_output):
        # Parsing out the markdown fencing
        lines = json_output.splitlines()
        for i, line in enumerate(lines):
            if line == "```json":
                json_output = "\n".join(lines[i+1:])  # Remove everything before "```json"
                json_output = json_output.split("```")[0]  # Remove everything after the closing "```"
                break  # Exit the loop once "```json" is found
        return json_output
    
    def locate_private_content(self, img_full_path, prompt, temperature = 1, top_p = 0.8, grounding = True, max_token = 128):
    
        max_new_tokens = max_token
        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant"
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    },
                    {
                      "image": img_full_path
                    }
                ]
            }
        ]
    
        if isinstance(img_full_path, Image.Image):
            image = img_full_path
        else:
            image = Image.open(img_full_path)
            
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=[image], padding=True, return_tensors="pt").to('cuda')
    
        output_ids = self.model.generate(**inputs, max_new_tokens=max_token)#, temperature= temperature, top_p = top_p)
        generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
        output_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    
        input_height = inputs['image_grid_thw'][0][1]*14
        input_width = inputs['image_grid_thw'][0][2]*14
        response = output_text[0]
        #print(response)
    
        if grounding == True:
            image_width, image_height = image.size
            image.thumbnail([1000, 1000], Image.Resampling.LANCZOS)
    
            if 'bbox_2d' in response:
                image, bboxes, labels = self.plot_bounding_boxes(image,response,input_width,input_height,image_width, image_height)
            else:
                image = image
                bboxes = None
                labels = None
    
        else:
            image  = image
            bboxes = None
            labels = None
    
        return response, image, bboxes, labels


    def labels_private_content(self, prompt, temperature = 1, top_p = 0.8, max_token=256):
    
        max_new_tokens=max_token
        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant"
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            }
        ]
            
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], return_tensors="pt").to('cuda')
    
        output_ids = self.model.generate(**inputs, max_new_tokens=max_token)#, temperature= temperature, top_p = top_p)
        generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
        output_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    
        response = output_text[0]
        json = self.parse_json(response)
        topics = ast.literal_eval(json)
        return response, topics
    
    def detect_rotation(self, cropped_image, plausible_rotations):
        
        yes_probs = []
    
        for rotation_ii in plausible_rotations:
            cropped_image_tmp = cropped_image.copy()
            #rotated_image = cropped_image_tmp.rotate(rotation_ii)
    
            _, rotated_image = self.rotate_image(pil_to_opencv(cropped_image_tmp), rotation_ii, 'rotated.jpg')
    
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
    
            # Preparation for inference
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to(self.model.device)
    
            # Inference: Generation of the output
            with torch.no_grad():
                outputs = self.model.generate(**inputs, max_new_tokens=128,
                                      return_dict_in_generate=True,
                                      use_cache=False,
                                      # return_dict=True,
                                      output_scores=True,
                                      do_sample=False)
                generated_ids = outputs["sequences"]
                generated_ids_trimmed = [
                    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                output_text = self.processor.batch_decode(
                    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )
    
                inputs_for_prob = self.processor(text=["Yes"],
                            images=image_inputs,
                            videos=video_inputs,
                            padding=True,
                            return_tensors="pt")
                yes_id = inputs_for_prob['input_ids'][0][0].item()
        
                inputs_for_prob = self.processor(text=["No"],
                                images=image_inputs,
                                videos=video_inputs,
                                padding=True,
                                return_tensors="pt")
                no_id = inputs_for_prob['input_ids'][0][0].item()
    
                logits = outputs["scores"][0][0]
                probs = (torch.nn.functional.softmax(torch.tensor([logits[yes_id], logits[no_id]]), dim=0).detach().cpu().numpy())
                print(f"Output text: {output_text} :: {rotation_ii} - Yes prob: {probs[0]}, No prob: {probs[1]}")
                yes_probs.append(probs[0])

        # get two max scores:
        max_idxs = sorted(range(len(yes_probs)), key=lambda i: yes_probs[i], reverse=True)[:2]
    
        return plausible_rotations[max_idxs[0]]
    
    def get_pseudo_detections(self, img_full_path, threshold_conf):

        ans, img, bboxes, labels = self.locate_private_content(img_full_path, "Spotting all the text and provide a label in the image with line-level, and output in JSON format.", temperature = 1, top_p=0.8)#, show_results = False)
    
        return img, bboxes, labels
    
    def warp_bboxes(self, bboxes, matrix):
    
        n          = bboxes.shape[0]
        
        for i in range(n):
            m = bboxes[i].shape[0]
            for j in range(m):
                
                detection = bboxes[i,j,:]
                h1 = matrix @ np.array([detection[0], detection[1], 1])
                bboxes[i,j,:] = h1[0:2]/h1[2]
    
        return bboxes
    
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
        #return M_I[0:2,:], opencv_to_pil(rotated.astype(np.uint8))
    
    def nms(self, bboxes, scores, labels, iou_threshold=0.5):
        """
        Perform Non-Maximum Suppression (NMS) on bounding boxes.
        
        Parameters:
        - bboxes: numpy array of shape (N, 4), where each row is (x1, y1, x2, y2)
        - scores: numpy array of shape (N,), confidence scores of each bbox
        - iou_threshold: float, IoU threshold for suppression
        
        Returns:
        - indices of the remaining bounding boxes after NMS
        """
        if len(bboxes) == 0:
            return [], [], []
    
        # Convert to numpy arrays if not already
        bboxes = np.array(bboxes)
        scores = np.array(scores)
    
        # Compute area of each bounding box
    
        poly = Polygon(bboxes[0])
        
        areas = [Polygon(matrix).area for matrix in bboxes]
        areas = np.array(areas)
        
        # Sort boxes by confidence score in descending order
        order = np.argsort(scores)[::-1]
    
        keep = []
        while order.size > 0:
            # Pick the bbox with the highest score and remove it from order
            idx = order[0]
            keep.append(idx)
    
            main_poly = Polygon(bboxes[idx])
            polygons  = bboxes[order[1:]]
            
            polygon_list = [Polygon(matrix) for matrix in polygons]
            intersection = [main_poly.intersection(poly) for poly in polygon_list]
            intersection = [inter.area if not inter.is_empty else 0 for inter in intersection]
    
            intersection = np.array(intersection)
            
            # Compute IoU
            iou = intersection / (areas[idx] + areas[order[1:]] - intersection)
    
            # Keep boxes with IoU less than the threshold
            order = order[1:][iou < iou_threshold]
    
        return bboxes[keep,:], scores[keep], [labels[k] for k in keep ]#labels[np.array(keep),:]
    
    def refine_text(self, image, new_bboxes, new_texts):
    
        cv_image  = pil_to_opencv(image)
        new_image = np.zeros(cv_image.shape).astype(np.uint8)
        
        for id_box, (box, label) in enumerate(zip(new_bboxes, new_texts)):
                    
            zeros = np.zeros_like(cv_image)
            mask  = cv2.fillPoly(zeros, [box.astype(np.int32)], color=(255, 255, 255))
            mask  = mask.astype(bool)
            new_image = mask * cv_image #cv_image = cv2.fillPoly(cv_image, [box.astype(np.int32)], color=(255, 255, 255))
            main_poly = Polygon(box)

            ans, _, _,_  = self.locate_private_content(opencv_to_pil(new_image), "Please output only the text content from the image without any additional descriptions or formatting.", temperature = 1, top_p=0.8, grounding = False)
    
            if len(ans.split(" ")) == 1:
                new_texts[id_box] = ans    
    
        return new_texts
    
    def rotation_fine_grained(self, image, mask, find_rotation=False, mode = 'OCR'):
    
        rotations = np.arange(0,360,10)
        max_distancia, min_distancia = -1e9, 1e9
        new_bboxes = []#np.empty((0,4))#np.array([])
        new_texts  = []
        new_confs  = []
    
        new_image = None
    
        if find_rotation:
            rotate = self.detect_rotation(image, np.array([0,90,180,270]))
            rotations = [rotate]
        else:
            rotations = [0,90,180,270]
        
    
        image = pil_to_opencv(image)
        image = image * np.stack([mask, mask, mask], axis=-1)
        for rotation in rotations:#[0, 45, 90, 135, 180, 225, 270, 315]:#rotations:
    
            image_file = 'rotated.jpg'
            matrix, rotated_image = self.rotate_image(image, rotation, image_file)
    
            bboxes, texts, elapse = text_detector(pil_to_opencv(rotated_image))
            confs = np.ones(bboxes.shape[0])
            
            temp_bboxes = self.warp_bboxes(bboxes, matrix) 
            for id_box, (box, text, conf) in enumerate(zip(temp_bboxes, texts, confs)):
            
                main_poly = Polygon(box)
                if explain_validity(main_poly) == 'Valid Geometry':
                    new_bboxes.append(box)
                    new_texts.append(text)
                    new_confs.append(conf)
    
        new_bboxes, _, new_texts = self.nms(new_bboxes, new_confs, new_texts, iou_threshold=0.5)
    
        if len(new_bboxes) <= 3:
            return [], [], [], []
                    
        return new_image, np.array(new_bboxes), new_texts, rotations
        

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.generic):  # Handle any NumPy type (e.g., int64, float64)
            return obj.item()  # Convert NumPy type to native Python type
        return super().default(obj)


def image_to_sam(image, predictor, p_boxes):
    
    predictor.set_image(pil_to_opencv(image))
    bounding_boxes = np.array([int(p_boxes[0]),int(p_boxes[1]),int(p_boxes[2]),int(p_boxes[3])])
    masks, _, _ = predictor.predict(point_coords=None, point_labels=None, box=bounding_boxes[None,:], multimask_output=False)
    y,x = np.where(masks[0]==True)
    thresh = (masks[0]*255).astype(np.uint8)#['segmentation'].astype(np.uint8)
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)
    segmentation = []
    for s_contours in sorted_contours:
        segmentation.append(list(s_contours.flatten()))

    return masks, y, x, segmentation
        
def extract_guarantee(text):
    match = re.search(r"\w*:\s*(.+)", text)
    if match:
        return match.group(1).strip()
    return 'Not Sure'


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--path")
    parser.add_argument('--do_rotation', action='store_true', help="Set this flag to True")

    args2 = parser.parse_args()


    folder_path = args2.path # "/gpfs/projects/CascanteBonillaGroup/datasets/BIV-Priv_Image/support_images"

    device = "cuda"

    sam_checkpoint = "../sam_vit_h_4b8939.pth"
    model_type = "vit_h"

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)

    qwen = QWEN_VLM()
   
    files = glob.glob(folder_path + "/*")
    #print(files)
    #files = [files[0]]#, files[2], files[8]]#, files[2]]#[files[1],files[2],files[3],files[4], files[5], files[6],files[7], files[8]]
    
    predictor = SamPredictor(sam)
    
    detection_info = {}
    
    for ind, file in enumerate(files):
    
        file_name = file.split('/')[-1]
        detection_info[file_name] = {'crop': [], 'segmentation': [], 'rotation': [], 'category': [], 'fine_grained': [], 'text':[], 'label': []}
        
        image = Image.open(file)
        
        

        ans, img, bboxes, _ = qwen.locate_private_content(image.copy(), "Locate paper document in the image, and output in JSON format.", temperature = 1, top_p=0.8)#, show_results = False)
        if bboxes is None:
            bboxes = []
        for p_boxes in bboxes:
    
            masks, y, x, segmentation = image_to_sam(image, predictor, p_boxes)

            
            crop_x1,crop_y1,crop_x2,crop_y2  = int(np.min(x)), int(np.min(y)), int(np.max(x)), int(np.max(y))
            crop_image = image.copy().crop([crop_x1,crop_y1,crop_x2,crop_y2])

            category = qwen.guided_labeling(crop_image)
            category = extract_guarantee(category)
    
            boxes = []

            fine_grained, texts, labels, rotations = [], [], [], []
            w, h = crop_image.size
            
            if w > 28 and h > 28:
    
                mask = masks[0]
                crop_mask = mask[crop_y1:crop_y2,crop_x1:crop_x2]
                _, boxes, d_texts, rots = qwen.rotation_fine_grained(crop_image, crop_mask, args2.do_rotation) 

    
                if len(boxes) != 0:
                    boxes[:,:,0] += crop_x1
                    boxes[:,:,1] += crop_y1
            
            if len(boxes) == 0:
    
                image = pil_to_opencv(image)
                image[masks[0]] = [0,0,0]
                index = str(ind)
                category = 'unsure'

            else:
    
                image = pil_to_opencv(image)
                for box, text in zip(boxes,d_texts):
                    
                    l_box = list(box.flatten().astype(int))
                    fine_grained.append(l_box)
                    texts.append(text)
                    labels.append('dummy_label')
                    cv2.fillPoly(image, [box.astype(np.int32)], color=(0, 0, 0))
  
            print(category)
            detection_info[file_name]['category'].append(category)
            detection_info[file_name]['segmentation'].append(segmentation)
            detection_info[file_name]['crop'].append([crop_x1,crop_y1,crop_x2,crop_y2])
            detection_info[file_name]['fine_grained'].append(fine_grained)
            detection_info[file_name]['text'].append(texts)
            detection_info[file_name]['label'].append(labels)
            detection_info[file_name]['rotation'].append(rotations)
                      
                
   
    with open("detection_info.json", "w") as f:
        json.dump(detection_info, f, cls=NumpyEncoder)
    
if __name__ == '__main__':
    main()
