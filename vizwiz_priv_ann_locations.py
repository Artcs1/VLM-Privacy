import os
import sys

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3"
os.environ['PYOPENGL_PLATFORM'] = 'egl'  # or try 'osmesa'

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

import trimesh
import pyrender
import pickle
        
def inpaint_3d_object(model_path, object_scale_factor = 0.5, x_rotation = 55, y_rotation = 120):
    # 1. Load your Trimesh scene (which has a camera)
    scene = trimesh.load(model_path)  # or .obj, etc.

    # If the loaded object is just a mesh, wrap it in a scene so we can have a camera
    if isinstance(scene, trimesh.Trimesh):
        scene = trimesh.Scene(scene)

    # scene.camera -> <trimesh.scene.Camera>
    # scene.camera_transform -> 4x4 matrix with camera extrinsics
    trimesh_camera = scene.camera
    trimesh_camera_transform = scene.camera_transform

    # print('Trimesh camera:', trimesh_camera)
    # Example output:
    # <trimesh.scene.Camera>
    #  FOV: [60. 45.]
    #  Resolution: [1800 1350]

    # 2. Convert the Trimesh scene to a PyRender scene
    pyrender_scene = pyrender.Scene.from_trimesh_scene(scene)

    # ====

    # Identify one of the mesh nodes to rotate.
    # (If your scene contains multiple nodes, choose the one you want to rotate.)
    mesh_node = None
    for node in pyrender_scene.nodes:
        if node.mesh is not None:
            mesh_node = node
            break

    if mesh_node is None:
        raise ValueError("No mesh node found in the scene!")

    # Create a rotation matrix: rotate 45° (pi/4 radians) about the Y-axis.
    angle = np.radians(y_rotation)
    rotation1 = trimesh.transformations.rotation_matrix(angle, [0, 1, 0])
    # Apply the rotation to the mesh node.
    # Multiplying on the left applies the rotation before the node's current transform.
    new_pose = rotation1 @ mesh_node.matrix
    pyrender_scene.set_pose(mesh_node, pose=new_pose)

    angle = np.radians(x_rotation)
    # Create a rotation matrix: rotate 10° (pi/4 radians) about the X-axis.
    rotation2 = trimesh.transformations.rotation_matrix(angle, [1, 0, 0])
    new_pose = rotation2 @ mesh_node.matrix
    pyrender_scene.set_pose(mesh_node, pose=new_pose)

    # Define your scale factor (e.g., 0.5 will make the object half as large)
    # scale_factor = 0.5
    # Create a 4x4 scaling matrix
    scale_matrix = np.array([
        [object_scale_factor, 0,             0,             0],
        [0,            object_scale_factor,  0,             0],
        [0,            0,             object_scale_factor,  0],
        [0,            0,             0,             1]
    ])
    # Apply the scaling to the current transformation of the mesh node.
    # If the object is centered at its origin, this will uniformly scale it.
    new_pose = scale_matrix @ mesh_node.matrix
    pyrender_scene.set_pose(mesh_node, pose=new_pose)
    # ====

    # 3. Create a matching PyRender camera
    # Trimesh stores FOV in degrees as [fov_x, fov_y], and PyRender's PerspectiveCamera
    # expects yfov in RADIANS. We'll assume fov_y is the vertical FOV.
    fov_y_degrees = trimesh_camera.fov[1]
    yfov_radians = np.radians(fov_y_degrees)

    # Create the PyRender camera with that vertical FOV
    camera = pyrender.PerspectiveCamera(yfov=yfov_radians)

    # 4. Add the camera to the PyRender scene at the same transform
    # Note: scene.camera_transform is a 4x4 matrix in world coordinates
    camera_node = pyrender_scene.add(camera, pose=trimesh_camera_transform)

    # ===

    # How far to move the camera (adjust this value as needed)
    offset = 0.1
    # Create a translation matrix that moves along the local z-axis.
    # Multiplying on the right applies the translation in the camera's local coordinate system.
    translation = np.eye(4)
    translation[2, 3] = offset  # Move along local z
    # Compute the new camera transform by applying the translation.
    new_camera_transform = trimesh_camera_transform @ translation
    # Update the camera node's pose.
    pyrender_scene.set_pose(camera_node, pose=new_camera_transform)

    # ===

    # 5. Add a light so the object is visible
    light = pyrender.DirectionalLight(color=np.ones(3), intensity=3.0)
    pyrender_scene.add(light, pose=trimesh_camera_transform)

    # 6. Use the same resolution for the offscreen renderer if you want a 1:1 match
    width, height = trimesh_camera.resolution
    renderer = pyrender.OffscreenRenderer(viewport_width=width, viewport_height=height)

    # 7. Render
    color, depth = renderer.render(pyrender_scene)
    renderer.delete()

    # 8. Save or display the resulting image
    # Image.fromarray(color).save('rendered_image.png')
    # plt.imshow(Image.fromarray(color))
    # plt.show()

    # ====================

    # Convert the rendered image to a PIL Image with an alpha channel
    rendered_img = Image.fromarray(color).convert("RGBA")

    # Optional: Make a simple white-to-transparent conversion
    # Adjust the threshold (here 240) depending on your render background color
    pixels = rendered_img.getdata()
    new_pixels = []
    for pixel in pixels:
        # if pixel is nearly white, set alpha to 0 (transparent)
        if pixel[0] > 240 and pixel[1] > 240 and pixel[2] > 240:
            new_pixels.append((255, 255, 255, 0))
        else:
            new_pixels.append(pixel)
    rendered_img.putdata(new_pixels)

    # ====================
    # ====================

    # Extract the alpha channel from the rendered image
    alpha = rendered_img.split()[3]
    # Get the bounding box of all non-zero (non-transparent) pixels
    bbox = alpha.getbbox()
    if bbox:
        # Crop the image to the bounding box
        cropped_img = rendered_img.crop(bbox)
        # plt.imshow(cropped_img)
        # plt.show()

    return cropped_img
        
 

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

        folder_path = "hot3d_dataset/object_models"
        files = glob.glob(folder_path + "/*.glb")
        
        files.sort()
        files[0]
        
        all_3d_objects = {}
        
        for ii in range (len(files)):
            info = {}
            if ii < len(files) - 1:
                info['rendered_image'] = inpaint_3d_object(files[ii], object_scale_factor = 0.5, x_rotation = 55, y_rotation = 210)
                info['object_scale_factor'] = 0.5
                info['x_rotation'] = 55
                info['y_rotation'] = 210
            else: # last object is weird //  a black control
                info['rendered_image'] = inpaint_3d_object(files[ii], object_scale_factor = 0.5, x_rotation = 0, y_rotation = 330)
                info['object_scale_factor'] = 0.5
                info['x_rotation'] = 0
                info['y_rotation'] = 330
            all_3d_objects[files[ii]] = info
        
        with open('results_qwen_72B_img_categories/3d_objects.pkl', 'wb') as file:
            pickle.dump(all_3d_objects, file)
        
        with open('results_qwen_72B_img_categories/all_meta_categories_locations_v3.json', 'r') as file:
            data = json.load(file)

        with open('results_qwen_72B_img_categories/3d_objects.pkl', 'rb') as file:
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
