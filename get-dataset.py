#!/usr/bin/env python3
import os
import shutil
from argparse import ArgumentParser

def download_dataset(username, key):
    os.environ['KAGGLE_USERNAME'] = username
    os.environ['KAGGLE_KEY'] = key


    import kaggle
    kaggle.api.dataset_download_files('nikhilpandey360/chest-xray-masks-and-labels', path='.', unzip=True)

    shutil.rmtree("Lung Segmentation")
    os.rename("data/Lung Segmentation", "data/LungSegmentation")

    DIR = "data/LungSegmentation/masks"
    all_masks = os.listdir(DIR)
    for m in all_masks:
        if "mask" not in m:
            new_name = m[0:-4]+"_mask.png"
            shutil.move(os.path.join(DIR, m), os.path.join(DIR, new_name))

def main():
    parser = ArgumentParser(description="Kaggle dataset downloader")
    parser.add_argument("--username", "-u", metavar="STR", type=str, help="Kaggle API username", required=True)
    parser.add_argument("--key", "-k", metavar="STR", type=str, help="Kaggle API key", required=True)

    args = parser.parse_args()
    
    download_dataset(args.username, args.key)


if __name__ == "__main__":
    main()
