import os

import torch
import re
import json
import pickle
import matplotlib.pyplot as plt
import numpy as np
import base64
import io
import cv2 
import base64
import html
import argparse

from tqdm import tqdm
from PIL import Image, ImageDraw
from io import BytesIO
from qwen_vl_utils import process_vision_info
from utils import *
from tqdm import tqdm

def main():

    parser = argparse.ArgumentParser(description="VizWiz VQA Full Inference.")
    parser.add_argument('-c', '--control_object', type=str, help='Control Object to use', default='mouse', 
                        choices=['mouse', 'wooden spoon', 'puzzle toy', 'red spatula', 'dino toy', 'milk carton', 'keyboard', 'dumbbell', 'remote control', 'whiteboard marker'])
    parser.add_argument('-m', '--model', type=str, help='Model to use', default='qwen25', 
                        choices=['qwen25', 'llava16', 'gemma3'])
    args = parser.parse_args()

    if not os.path.exists(f"results_{args.model}_7B_img_categories_v2"):
        os.makedirs(f"results_{args.model}_7B_img_categories_v2")

    with open('results_qwen_72B_img_categories/all_meta_categories_locations_v4.json', 'r') as file:
        data = json.load(file)

    with open('results_qwen_72B_img_categories/3d_objects.pkl', 'rb') as file:
        all_3d_objects = pickle.load(file)

    with open('hot3d_dataset/object_models/models_info.json', 'r') as file:
        all_3d_objects_names = json.load(file)          

    meta_categories = [
    "bank statement",
    "letter with address",
    "credit or debit card",
    "bills or receipt",
    "preganancy test",
    "pregnancy test box",
    "mortage or investment report",
    "doctor prescription",
    "empty pill bottle",
    "condom with plastic bag",
    "tattoo sleeve",
    "transcript",
    "business card",
    "condom box",
    "local newspaper",
    "medical record document",
    "email",
    "phone",
    "id card",
    ]


    answerable_questions = {
        "bank statement": [
            "What does this say?",
            "I need to identify the kind of form I have here. Can you provide me with just a few words that you can read on this piece of paper?",
            "Who is this from?",
            "What are the amounts?",
            "Read the statement."
        ],
        "letter with address": [
            "What does this say?",
            "Is this piece of mail?",
            "Can you tell me who this is from?",
            "I just wanted to know who the return address is on the, or what the return address is on this envelope on the upper left hand corner.",
            "What is this mail, where is it from?",
            "Who is this mail for?"
        ],
        "credit or debit card": [
            "What is this?",
            "What kind of card is this?",
            "What is the expiration date?",
            "Is there a phone number on this card and what is it?",
            "Can you please tell me the 1-800 number on this card?",
            "Can you read this card number?"
        ],
        "bills or receipt": [
            "What does this say?",
            "What bill is this?",
            "How much is this bill for?",
            "What is the total amount?",
            "I know that this is a receipt, but what is it a receipt of?"
        ],
        "pregnancy test": [],
        "pregnancy test box": [
            "What does this say?",
            "What is the expiration date?"
        ],
        "mortage or investment report": [
            "What does this say?",
            "I need to identify the kind of form I have here. Can you provide me with just a few words that you can read on this piece of paper?",
            "Who is this from?"
        ],
        "doctor prescription": [
            "What does this say?",
            "Can you read the name of this prescription?",
            "What kind of medications?",
            "What are the instructions?",
            "What's the side effects?"
        ],
        "empty pill bottle": [
            "What does this say?",
            "Can you read what kind of medicine is in the bottle? If you can only tell me what letter it starts with I'll know the rest.",
            "What kind of medication is it?",
            "What are the instructions on this bottle?",
            "What's the side effects on this medicine bottle?"
        ],
        "condom with plastic bag": [],
        "tattoo sleeve": [],
        "transcript": [
            "What does this say?"
        ],
        "business card": [
            "What does this say?",
            "What does this business card have on it?",
            "What is the card in my hand?",
            "What is the phone number on this business card?\nCan you please tell me who this business card is from?\nWhat does this say? I'm looking for a phone number and email address. Thank you."
        ],
        "condom box": [
            "What is this?",
            "What is the expiration date?"
        ],
        "local newspaper": [
            "What does this say?",
            "Can you please describe what's on this newspaper?",
            "What's the date of this paper?",
            "What is this newspaper section about?",
            "What newspaper is this?"
        ],
        "medical record document": [
            "What does this say?",
            "I need to identify the kind of form I have here. Can you provide me with just a few words that you can read on this piece of paper?",
            "Who is this from?"
        ],
        "email": [],
        "phone": [],
        "id card": [
            "What is this?",
            "What kind of card is this?",
            "What is the phone number on this card and what is it?"
        ]
    }

    object_name_to_id = {'black holder': '01',
                        'bowl': '02',
                        'plate': '03',
                        'wooden spoon': '04',
                        'potato masher': '05',
                        'red spatula': '06',
                        'coffee pot': '07',
                        'patterned mug': '08',
                        'white mug': '09',
                        'can of soup': '10',
                        'can of parmesan': '11',
                        'can of tomato sauce': '12',
                        'mustard bottle': '13',
                        'bbq bottle': '14',
                        'ranch bottle': '15',
                        'vase': '16',
                        'milk carton': '17',
                        'carton': '18',
                        'flask': '19',
                        'waffles': '20',
                        'vegetables': '21',
                        'dumbbell': '22',
                        'small aria': '23',
                        'cellphone': '24',
                        'gray holder': '25',
                        'small birdhouse': '26',
                        'dino toy': '27',
                        'keyboard': '28',
                        'whiteboard eraser': '29',
                        'puzzle toy': '30',
                        'mouse': '31',
                        'whiteboard marker': '32',
                        'remote control': '33'}
    
    with open(f'results_qwen_72B_img_categories/documents_fine_grained_labels_per_metacategory.json', 'r') as file:
        documents_data_finegrained_categories = json.load(file)

    with open(f'results_qwen_72B_img_categories/creditcards_fine_grained_labels_per_metacategory.json', 'r') as file:
        creditcards_data_finegrained_categories = json.load(file)

    if args.model == "qwen25":
        model, processor = qwen25()
    elif args.model == "llava16":
        model, processor = llava16()
    elif args.model == "gemma3":
        model, processor = gemma3()

    def ask_extra_answerable_questions(image_to_ask, anns):
        answerable_qas = []
        if anns['label'] not in answerable_questions.keys():
            return answerable_qas
        
        for q in answerable_questions[anns['label']]:
            answerable = f"Given this question: '{q}', based on the image, are you able to answer the question?"
            answerable_score = vqa_yes_prob(model, processor, image_to_ask, f"Is there a {answerable} in this image?", model=args.model)
            answerable_qas.append([answerable, answerable_score])
        return answerable_qas

    print(len(data))

    

    for ii, (img_path, anns) in tqdm(enumerate(data.items())):


        if anns['full_image_mask'] != 'None': # and anns['high_risk_fine_grained_masked_private_obj'] != 'None':

            # fix object to mouse
            object_number = object_name_to_id[args.control_object] # 31 # extract_number(anns['object_path'])
            object_path = f'hot3d_dataset/object_models/obj_0000{object_number}.glb'
            object_name = all_3d_objects_names[str(int(object_number))]['name']
            obj = all_3d_objects[object_path]['rendered_image'] # all_3d_objects[anns['object_path']]['rendered_image']
            
            full_image = Image.open(img_path)
            full_image.paste(obj, anns['object_position'][:2], obj)

            # plt.imshow(full_image)
            # plt.show()

            full_mask_vis = base64_str_to_image2(anns['full_image_mask'])
            full_mask_vis[full_mask_vis==1]=255
            full_mask_vis = Image.fromarray(full_mask_vis).convert("L")
            full_mask_vis = np.array(full_mask_vis).astype(bool)
            
            masked_full_image = full_image_zero_outside_mask(img_path, ~full_mask_vis)
            masked_full_image.paste(obj, anns['object_position'][:2], obj)

            # plt.imshow(masked_full_image)
            # plt.show()

            # ========================================================================

            file_name = os.path.basename(img_path).replace('.jpeg', '.json')

            if os.path.exists(f'results_qwen_72B_creditcards_augmented/{file_name}'):
                with open(f'results_qwen_72B_creditcards_augmented/{file_name}', 'r') as file:
                    data_finegrained = json.load(file)
                with open(f'results_qwen_72B_creditcards_labels/{file_name}', 'r') as file:
                    data_finegrained_labels = json.load(file)  
                try:
                    data_finegrained_categories = creditcards_data_finegrained_categories[img_path]['finegrained_labels']
                except:
                    data_finegrained_categories = []
            else:
                # documents fine-grained
                with open(f'results_qwen_72B_augmented/{file_name}', 'r') as file:
                    data_finegrained = json.load(file)
                with open(f'results_qwen_72B_labels/{file_name}', 'r') as file:
                    data_finegrained_labels = json.load(file)  
                try:
                    data_finegrained_categories = documents_data_finegrained_categories[img_path]['finegrained_labels']
                except:
                    data_finegrained_categories = []


            all_fine_grained_masked = Image.open(img_path)
            draw_all = ImageDraw.Draw(all_fine_grained_masked)
            for poly in data_finegrained['original_detailed_data'][0]['data_vlm'][0]:
                # Convert each coordinate pair to integer values
                points = [(int(round(x)), int(round(y))) for x, y in poly]
                draw_all.polygon(points, fill="black")  # Change fill color if needed

            all_fine_grained_masked.paste(obj, anns['object_position'][:2], obj)

            # ===========================================================================================

            # if data_finegrained_labels['labels_full_image'] != 'Empty':

               
            if data_finegrained_categories and data_finegrained['original_detailed_data'][0]['data_vlm'][0]:
                
                high_risk_fine_grained_masked = Image.open(img_path)
                draw_high_risk = ImageDraw.Draw(high_risk_fine_grained_masked)

                for ii_text, text in enumerate(data_finegrained['original_detailed_data'][0]['data_vlm'][1]):
                    
                    if data_finegrained_categories[text] not in ['none', 'other']:
                        poly = data_finegrained['original_detailed_data'][0]['data_vlm'][0][ii_text]
                        points = [(int(round(x)), int(round(y))) for x, y in poly]
                        draw_high_risk.polygon(points, fill="black")
                        high_risk_fine_grained_present = True

                high_risk_fine_grained_masked.paste(obj, anns['object_position'][:2], obj)

            # ===========================================================================================
            # ===========================================================================================

            # print (anns['label'], '-', object_name)

 #           plt.imshow(full_image)
 #           # plt.show()
 #           plt.savefig("full_image.png")


 #           plt.imshow(masked_full_image)
 #           # plt.show()
 #           plt.savefig("masked_full_image.png")

 #           plt.imshow(all_fine_grained_masked)
 #           # plt.show()
 #           plt.savefig("fine_full_image.png")

 #           if high_risk_fine_grained_masked:
 #               plt.imshow(high_risk_fine_grained_masked)
 #               # plt.show()
 #               plt.savefig("high_risk.png")
 #


            full_image_private_obj = vqa_yes_prob(model, processor, full_image, f"Is there a {anns['label']} in this image?", model=args.model)
            full_image_non_private_obj = vqa_yes_prob(model, processor, full_image, f"Is there a {object_name} in this image?", model=args.model)

            anns['full_image_private_obj'] = full_image_private_obj
            anns['full_image_non_private_obj'] = full_image_non_private_obj
            # print (full_image_private_obj, '-', full_image_non_private_obj)
            answerable_qas = ask_extra_answerable_questions(full_image, anns)
            anns['full_image_answerable_qas'] = answerable_qas

            masked_full_image_private_obj = vqa_yes_prob(model, processor, masked_full_image, f"Is there a {anns['label']} in this image?", model=args.model)
            masked_full_image_non_private_obj = vqa_yes_prob(model, processor, masked_full_image, f"Is there a {object_name} in this image?", model=args.model)
            anns['masked_full_image_private_obj'] = masked_full_image_private_obj
            anns['masked_full_image_non_private_obj'] = masked_full_image_non_private_obj
            # print (masked_full_image_private_obj, '-', masked_full_image_non_private_obj)
            answerable_qas = ask_extra_answerable_questions(masked_full_image, anns)
            anns['masked_full_image_answerable_qas'] = answerable_qas

            all_fine_grained_masked_private_obj = vqa_yes_prob(model, processor, all_fine_grained_masked, f"Is there a {anns['label']} in this image?", model=args.model)
            all_fine_grained_masked_non_private_obj = vqa_yes_prob(model, processor, all_fine_grained_masked, f"Is there a {object_name} in this image?", model=args.model)
            anns['all_fine_grained_masked_private_obj'] = all_fine_grained_masked_private_obj
            anns['all_fine_grained_masked_non_private_obj'] = all_fine_grained_masked_non_private_obj
            # print (all_fine_grained_masked_private_obj, '-', all_fine_grained_masked_non_private_obj)
            answerable_qas = ask_extra_answerable_questions(all_fine_grained_masked, anns)
            anns['all_fine_grained_masked_answerable_qas'] = answerable_qas

            # breakpoint()
            # plt.imshow(high_risk_fine_grained_masked)
            # plt.show()
            if high_risk_fine_grained_masked:
                plt.imshow(high_risk_fine_grained_masked)
                # plt.show()
                plt.savefig("high_risk.png")
 
                high_risk_fine_grained_masked_private_obj = vqa_yes_prob(model, processor, high_risk_fine_grained_masked, f"Is there a {anns['label']} in this image?", model=args.model)
                high_risk_fine_grained_masked_non_private_obj = vqa_yes_prob(model, processor, high_risk_fine_grained_masked, f"Is there a {object_name} in this image?", model=args.model)
                anns['high_risk_fine_grained_masked_private_obj'] = high_risk_fine_grained_masked_private_obj
                anns['high_risk_fine_grained_masked_non_private_obj'] = high_risk_fine_grained_masked_non_private_obj
                # print (high_risk_fine_grained_masked_private_obj, '-', high_risk_fine_grained_masked_non_private_obj)
                answerable_qas = ask_extra_answerable_questions(high_risk_fine_grained_masked, anns)
                anns['high_risk_fine_grained_masked_answerable_qas'] = answerable_qas
            else:
                # files_with_no_fine_grained.append([ii, img_path])
                anns['high_risk_fine_grained_masked_private_obj'] = 'None'
                anns['high_risk_fine_grained_masked_non_private_obj'] = 'None'
                anns['high_risk_fine_grained_masked_answerable_qas'] = 'None'
            # break
        else:
            # files_with_no_anns.append([ii, img_path])
            anns['all_fine_grained_masked_private_obj'] = 'None'
            anns['all_fine_grained_masked_non_private_obj'] = 'None'
            anns['high_risk_fine_grained_masked_private_obj'] = 'None'
            anns['high_risk_fine_grained_masked_non_private_obj'] = 'None'

        # break
        if ii % 100 == 0:
            filename = f"all_meta_categories_vqa_results_fixed_object_{args.control_object}.json"
            with open(f"results_{args.model}_7B_img_categories_v2/{filename}", "w") as fp:
                json.dump(data, fp, indent=4, cls=NumpyEncoder)

    filename = f"all_meta_categories_vqa_results_fixed_object_{args.control_object}.json"
    with open(f"results_{args.model}_7B_img_categories_v2/{filename}", "w") as fp:
        json.dump(data, fp, indent=4, cls=NumpyEncoder)
        
if __name__ == "__main__":
    main()
