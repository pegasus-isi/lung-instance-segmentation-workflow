#!/usr/bin/env python3
import sys
import argparse
import os
import cv2
import numpy as np
from cv2 import imread
import numpy as ppool 
import pdb
import random
import tensorflow as tf
import keras

class DataPreprocessing:  

    def readForTransformations(self, input_image, pth, is_mask):  
        img = cv2.imread(os.path.join(pth, input_image))
        if not is_mask: return cv2.resize(img,(X_shape,X_shape))[:,:,0] 
        return img 

    def flipImage(self, input_image, pth, is_mask=True):
        norm_img = np.zeros((800, 800))
        img = self.readForTransformations(input_image, pth, is_mask)
        if not is_mask: 
            flipped = cv2.flip(img, 1)
            return cv2.normalize(flipped,  norm_img, 0, 255, cv2.NORM_MINMAX)
        else:
            flipped = cv2.flip(img, 1)
            return flipped

    def rotateImage(self, input_image, pth, deg, is_mask = True):
        img = self.readForTransformations(input_image, pth, is_mask)
        h, w = img.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, deg, 1.0)
        rotated = cv2.warpAffine(img, M, (w, h))
        if not is_mask: 
            norm_img = np.zeros((800, 800))
            cv2.normalize(rotated,  norm_img, 0, 255, cv2.NORM_MINMAX)
        return rotated
    
    @staticmethod
    def normalize(input_image):
        """
        This function takes images as input and normalize it i.e remove noise from the data. By normalizing images, we bring
        the image into a range of intensity values which is normal to our senses. In a normalized image, the mean is 0 and the 
        variance is 1.        
        :param input_image: image_file[.png]
        :return:  image_file[.png]
        """
        norm_img = np.zeros((800, 800))
        im = cv2.resize(cv2.imread(os.path.join(DIR, input_image)),(X_shape,X_shape))[:,:,0]        
        return cv2.normalize(im,  norm_img, 0, 255, cv2.NORM_MINMAX)
    
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
    DIR = args.input_dir
    
    X_shape = 256 #converting the input images to 256x256 size
    norm_img = np.zeros((800,800))
    all_masks = []
    images = []
    for i in os.listdir(DIR):
        if ".png" in i: 
            if "mask" in i: all_masks.append(i)
            else: images.append(i)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    dp = DataPreprocessing()

    # Calling the normalize function to nomalize the images and saving them to the output directory
    for i in images: 
        img = i
        normalized_image1 = dp.normalize(i)
        cv2.imwrite(os.path.join(args.output_dir, args.type+'_'+i[0:-4]+"_norm.png"), normalized_image1)
        #creating transformations for the train images
        if args.type == "train": 
            mask_img = None 
            fname = img[:-4]
            for m in all_masks:
                mname = m[:-9]
                if fname == mname:
                    mask_img = m
                    break
            if mask_img is None: continue
            flipped = dp.flipImage(img, DIR, False)
            cv2.imwrite(os.path.join(args.output_dir, args.type+'_'+i[0:-4]+"_0_norm.png"), flipped)
            flipped = dp.flipImage(mask_img, args.output_dir)
            # print('Writing to ', os.path.join(args.output_dir, args.type+'_'+i[0:-4]+"_norm.png"))
            cv2.imwrite(os.path.join(args.output_dir, i[0:-4]+"_0_mask.png"), flipped)

            for ind, deg in enumerate([8]):
                rotated = dp.rotateImage(img, DIR, deg, False)
                cv2.imwrite(os.path.join(args.output_dir, args.type+'_'+i[0:-4]+"_"+str(ind+1)+"_norm.png"), rotated)
                # print('Writing to ',i[0:-4]+"_"+str(ind+1)+"_mask.png")
                rotated = dp.rotateImage(mask_img, args.output_dir, deg)
                cv2.imwrite(os.path.join(args.output_dir, i[0:-4]+"_"+str(ind+1)+"_mask.png"), rotated)