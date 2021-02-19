#!/usr/bin/env python3
import sys
import argparse
import os
import cv2
import numpy as np
from cv2 import imread 

class DataPreprocessing:    
    
    @staticmethod
    def normalize(input_image):
        """
        This function takes images as input and normalize it i.e remove noise from the data. By normalizing images, we bring
        the image into a range of intensity values which is normal to our senses. In a normalized image, the mean is 0 and the 
        variance is 1.        
        :param input_image: image_file[.png]
        :return:  image_file[.png]
        """
        
        im = cv2.resize(cv2.imread(os.path.join(DIR, input_image)),(X_shape,X_shape))[:,:,0]
        final_img = cv2.normalize(im,  norm_img, 0, 255, cv2.NORM_MINMAX)
        
        return final_img
    
def parse_args(args):
    """
    This function takes the command line arguments.        
    :param args: commandline arguments
    :return:  parsed commands
    """
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
    parser.add_argument(
                "-t",
                "--type",
                default="train",
                help="the type of input set stating if it is test/train/val"
            )

    return parser.parse_args(args)

if __name__=="__main__":
    args = parse_args(sys.argv[1:])
    DIR = os.path.join(args.input_dir)
    
    X_shape = 256 #converting the input images to 256x256 size
    norm_img = np.zeros((800,800))
    files = os.listdir(DIR)
    images = [i for i in files if ".png" in i]
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    dp = DataPreprocessing()

    # Calling the normalize function to nomalize the images and saving them to the output directory
    for i in images: 
        normalized_image = dp.normalize(i)
        cv2.imwrite(os.path.join(args.output_dir, args.type+'_'+i[0:-4]+"_norm.png"), normalized_image)

