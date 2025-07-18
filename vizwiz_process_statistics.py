import os
import re
import json
import pickle
import matplotlib.pyplot as plt
import numpy as np
import base64
import io
import glob

from PIL import Image, ImageDraw
from tqdm import tqdm


all_files = []
all_files.append('/gpfs/projects/CascanteBonillaGroup/paola/Qwen2.5-VL/results_qwen_7B_img_categories/all_meta_categories_vqa_results_fixed_object*.json')  # change to your paths
all_files.append('/gpfs/projects/CascanteBonillaGroup/paola/Qwen2.5-VL/results_llava16_7B_img_categories_v2/all_meta_categories_vqa_results_fixed_object*.json')  # change to your paths
all_files.append('/gpfs/projects/CascanteBonillaGroup/paola/Qwen2.5-VL/results_gemma3_7B_img_categories_v2/all_meta_categories_vqa_results_fixed_object*.json') # change to your paths

for path in all_files:
    
    print(path)
    files = glob.glob(path)

    total_records = 0
    yes_threshold = 0
    meta_scores = {'full_image_private_obj': [],
                    'full_image_non_private_obj': [],
                    'masked_full_image_private_obj': [],
                    'masked_full_image_non_private_obj': [],
                    'all_fine_grained_masked_private_obj': [],
                    'all_fine_grained_masked_non_private_obj': [],
                    'high_risk_fine_grained_masked_private_obj': [],
                    'high_risk_fine_grained_masked_non_private_obj': [] }
    
    for ii_file in files:
        #print (ii_file)
        with open(ii_file, 'r') as file:
            data = json.load(file)
    
        vqa_score_keys = [k for k in next(iter(data.values())) if '_private_obj' in k]
        grouped = {key: [entry[key] for entry in data.values() if entry['high_risk_fine_grained_masked_private_obj'] != 'None'] for key in vqa_score_keys}
    
        for key in vqa_score_keys:
            total_records = 0
            yes_threshold = 0
            for score_tmp in grouped[key]:
                if score_tmp[0] > 0.5: # 'yes' in score_tmp[0].lower():
                    yes_threshold += 1
                total_records += 1
    
            #print(key, ': ', yes_threshold/total_records)
            meta_scores[key].append(yes_threshold/total_records)
        #print ('===============')
    
    for ii_keys in vqa_score_keys:
        print (f"{ii_keys} : mean: {np.mean(meta_scores[ii_keys])} - std: {np.std(meta_scores[ii_keys])}")

    print('\n')

print('_____________________________________________________________________________________________')

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

path = '/gpfs/projects/CascanteBonillaGroup/paola/Qwen2.5-VL/results_qwen_7B_img_categories_v2/all_meta_categories_vqa_results_fixed_object*.json'
files = glob.glob(path)
all_tables = []

for ii_file in files:

    with open(ii_file, 'r') as file:
        data = json.load(file)
    answerable_temp_dict = {}
    table = {}
    
    for meta_cat_tmp in meta_categories:
        vqa_score_keys = [k for k in next(iter(data.values())) if '_answerable_qas' in k]
        grouped = {key: [entry[key] for entry in data.values() if entry['high_risk_fine_grained_masked_private_obj'] != 'None' and entry['label'] == meta_cat_tmp] for key in vqa_score_keys}
        if grouped['full_image_answerable_qas']:
            full_image_answerable_qas = grouped['full_image_answerable_qas']
            masked_full_image_answerable_qas = grouped['masked_full_image_answerable_qas']
            all_fine_grained_masked_answerable_qas = grouped['all_fine_grained_masked_answerable_qas']
            high_risk_fine_grained_masked_answerable_qas = grouped['high_risk_fine_grained_masked_answerable_qas']
            
            tam = len(full_image_answerable_qas[0])
           
            if tam != 0:
                table[meta_cat_tmp] = np.zeros((len(full_image_answerable_qas[0]),4))
    
            for questions in range(len(full_image_answerable_qas[0])):
                mean_score = np.mean([qas[questions][1][0] for qas in full_image_answerable_qas])
                table[meta_cat_tmp][questions, 0] = mean_score 
            for questions in range(len (masked_full_image_answerable_qas[0])):
                mean_score = np.mean([qas[questions][1][0] for qas in masked_full_image_answerable_qas])
                table[meta_cat_tmp][questions, 1] = mean_score 
            for questions in range(len (all_fine_grained_masked_answerable_qas[0])):
                mean_score = np.mean([qas[questions][1][0] for qas in all_fine_grained_masked_answerable_qas])
                table[meta_cat_tmp][questions, 2] = mean_score 
            for questions in range(len (high_risk_fine_grained_masked_answerable_qas[0])):
                mean_score = np.mean([qas[questions][1][0] for qas in high_risk_fine_grained_masked_answerable_qas])
                table[meta_cat_tmp][questions, 3] = mean_score 

    all_tables.append(table)

keys = meta_categories
tables = {key:np.mean([table[key] for table in all_tables], axis = 0) for key in keys if key in all_tables[0].keys()}

averages = {}
for table_name, data in tables.items():
    avg = np.mean(data, axis=0)
    averages[table_name] = avg
    print(f"{table_name}:")
    print(f"  Full Image:           {avg[0]:.4f}")
    print(f"  Masked Object:        {avg[1]:.4f}")
    print(f"  Fine-grained Masked:  {avg[2]:.4f}")
    print(f"  High-risk Masked:     {avg[3]:.4f}")
    print()
    
# Create a grouped bar plot comparing the averages.
table_names = list(tables.keys())
table_names_tmp = [tn.replace(" ", "\n") for tn in table_names]
n_tables = len(table_names)
x = np.arange(n_tables)  # the label locations
width = 0.2              # the width of the bars
    
fig, ax = plt.subplots(figsize=(14, 7))
plt.rcParams.update({'font.size': 12})
    
# Define labels for the four columns.
column_labels = ["Full Image", "Object Mask", "Fine-grained Mask", "High-Risk Mask"]
color_list = ["steelblue", "indianred", "darkorange", "forestgreen"]
    
# Plot each column's averages for all tables.
for i, label in enumerate(column_labels):
    col_values = [averages[table][i] for table in table_names]
    ax.bar(x + i * width - 1.5 * width, col_values, width, label=label, color=color_list[i], alpha=0.8, edgecolor='dimgray')
    
# Set the labels and title.
ax.set_xticks(x)
ax.set_xticklabels(table_names_tmp) #, rotation=45, ha="right")
ax.set_ylabel("Mean Accuracy")
# ax.set_title("Average Column Values per Table")
legend = ax.legend(prop={'size': 14}, loc="best") #, bbox_to_anchor=(1, 1.01))
legend.get_frame().set_facecolor((1, 1, 1, 0.5))
    
plt.tight_layout()
plt.savefig("answerable_plot.pdf")
plt.show()
