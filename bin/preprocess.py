#!/usr/bin/env python3
import sys
import argparse
import os
import cv2
import numpy as np
from cv2 import imread 
from pathlib import Path

class DataPreprocessing:    
    
    @staticmethod
    def normalize(i):
        
        im = cv2.resize(cv2.imread(os.path.join(DIR, i)),(X_shape,X_shape))[:,:,0]
        final_img = cv2.normalize(im,  norm_img, 0, 255, cv2.NORM_MINMAX)
        
        return final_img
    
def parse_args(args):
    parser = argparse.ArgumentParser(description="Preprocess Job - Normalizes the images for faster computation")
    parser.add_argument(
                "-i",
                "--input_dir",
                default=".",
                help="directory where input files will be read from"
            )

    parser.add_argument(
                "-o",
                "--output_dir",
                default=".",
                help="directory where output files will be written to"
            )

    return parser.parse_args(args)

if __name__=="__main__":
    args = parse_args(sys.argv[1:])
    DIR = args.input_dir
    X_shape = 256
    norm_img = np.zeros((800,800))
    files = os.listdir(DIR)
    images = [i for i in files if ".png" in i]

    dp = DataPreprocessing()

    # Calling the normalize function to nomalize the images and saving them to the output directory
    for i in images:        
        normalized_image = dp.normalize(i)
        cv2.imwrite(os.path.join(args.output_dir, i.split(".png")[0]+"_norm.png"), normalized_image)

