#!/usr/bin/env python3
import os
import kaggle
import shutil

os.environ['KAGGLE_USERNAME'] = "your_username"
os.environ['KAGGLE_KEY'] = "your_key"

kaggle.api.dataset_download_files('nikhilpandey360/chest-xray-masks-and-labels/download', path='.', unzip=True)

os.rename("data/Lung Segmentation", "data/LungSegmentation")

DIR = "data/LungSegmentation/masks"
all_masks = os.listdir(DIR)
for m in all_masks:
    if "mask" not in m:
        new_name = m[0:-4]+"_mask.png"
        shutil.move(os.path.join(DIR, m), os.path.join(DIR, new_name))