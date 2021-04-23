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

IGNORE_IMAGES = ['CHNCXR_0025_0.png', 'CHNCXR_0036_0.png', 'CHNCXR_0037_0.png', 'CHNCXR_0038_0.png', 'CHNCXR_0039_0.png', 'CHNCXR_0040_0.png', 'CHNCXR_0065_0.png', 'CHNCXR_0181_0.png', 'CHNCXR_0182_0.png', 'CHNCXR_0183_0.png', 'CHNCXR_0184_0.png', 'CHNCXR_0185_0.png', 'CHNCXR_0186_0.png', 'CHNCXR_0187_0.png', 'CHNCXR_0188_0.png', 'CHNCXR_0189_0.png', 'CHNCXR_0190_0.png', 'CHNCXR_0191_0.png', 'CHNCXR_0192_0.png', 'CHNCXR_0193_0.png', 'CHNCXR_0194_0.png', 'CHNCXR_0195_0.png', 'CHNCXR_0196_0.png', 'CHNCXR_0197_0.png', 'CHNCXR_0198_0.png', 'CHNCXR_0199_0.png', 'CHNCXR_0200_0.png', 'CHNCXR_0201_0.png', 'CHNCXR_0202_0.png', 'CHNCXR_0203_0.png', 'CHNCXR_0204_0.png', 'CHNCXR_0205_0.png', 'CHNCXR_0206_0.png', 'CHNCXR_0207_0.png', 'CHNCXR_0208_0.png', 'CHNCXR_0209_0.png', 'CHNCXR_0210_0.png', 'CHNCXR_0211_0.png', 'CHNCXR_0212_0.png', 'CHNCXR_0213_0.png', 'CHNCXR_0214_0.png', 'CHNCXR_0215_0.png', 'CHNCXR_0216_0.png', 'CHNCXR_0217_0.png', 'CHNCXR_0218_0.png', 'CHNCXR_0219_0.png', 'CHNCXR_0220_0.png', 'CHNCXR_0336_1.png', 'CHNCXR_0341_1.png', 'CHNCXR_0342_1.png', 'CHNCXR_0343_1.png', 'CHNCXR_0344_1.png', 'CHNCXR_0345_1.png', 'CHNCXR_0346_1.png', 'CHNCXR_0347_1.png', 'CHNCXR_0348_1.png', 'CHNCXR_0349_1.png', 'CHNCXR_0350_1.png', 'CHNCXR_0351_1.png', 'CHNCXR_0352_1.png', 'CHNCXR_0353_1.png', 'CHNCXR_0354_1.png', 'CHNCXR_0355_1.png', 'CHNCXR_0356_1.png', 'CHNCXR_0357_1.png', 'CHNCXR_0358_1.png', 'CHNCXR_0359_1.png', 'CHNCXR_0360_1.png', 'CHNCXR_0481_1.png', 'CHNCXR_0482_1.png', 'CHNCXR_0483_1.png', 'CHNCXR_0484_1.png', 'CHNCXR_0485_1.png', 'CHNCXR_0486_1.png', 'CHNCXR_0487_1.png', 'CHNCXR_0488_1.png', 'CHNCXR_0489_1.png', 'CHNCXR_0490_1.png', 'CHNCXR_0491_1.png', 'CHNCXR_0492_1.png', 'CHNCXR_0493_1.png', 'CHNCXR_0494_1.png', 'CHNCXR_0495_1.png', 'CHNCXR_0496_1.png', 'CHNCXR_0497_1.png', 'CHNCXR_0498_1.png', 'CHNCXR_0499_1.png', 'CHNCXR_0500_1.png', 'CHNCXR_0502_1.png', 'CHNCXR_0505_1.png', 'CHNCXR_0560_1.png', 'CHNCXR_0561_1.png', 'CHNCXR_0562_1.png', 'CHNCXR_0563_1.png', 'CHNCXR_0564_1.png', 'CHNCXR_0565_1.png']


class DataPreprocessing:  

    def readForTransformations(self, input_image, pth, is_mask):  
        if is_mask: 
            pth = "/local-work/aditi/lung-instance-segmentation-workflow/without-pegasus-orig/data/lung-masks"
        img = cv2.imread(os.path.join(pth, input_image))
        if not is_mask: return cv2.resize(img,(X_shape,X_shape))[:,:,0] 
        return img 

    def flipImage(self, input_image, pth, is_mask=True):
        norm_img = np.zeros((800, 800))
        img = self.readForTransformations(input_image, pth, is_mask)
        if not is_mask: 
            # img = cv2.resize(cv2.imread(os.path.join(pth, input_image)),(X_shape,X_shape))[:,:,0]  
            flipped = cv2.flip(img, 1)
            return cv2.normalize(flipped,  norm_img, 0, 255, cv2.NORM_MINMAX)
        else:
            # pth = "/local-work/aditi/lung-instance-segmentation-workflow/without-pegasus-orig/data/lung-masks"
            # img = cv2.imread(os.path.join(pth, input_image))
            flipped = cv2.flip(img, 1)
            return flipped

    def rotateImage(self, input_image, pth, deg, is_mask = True):
        img = self.readForTransformations(input_image, pth, is_mask)
        h, w = img.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, 10, 1.0)
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
    DIR = os.path.join(args.input_dir)
    mask_path = os.path.join("../data", "lung-masks")

    
    X_shape = 256 #converting the input images to 256x256 size
    norm_img = np.zeros((800,800))
    files = os.listdir(DIR)
    images = [i for i in files if ".png" in i and i not in IGNORE_IMAGES]
    masks = [i for i in os.listdir(mask_path) if ".png" in i]
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    dp = DataPreprocessing()

    # Calling the normalize function to nomalize the images and saving them to the output directory
    for i in images: 
        img = i
        normalized_image1 = dp.normalize(i)
        cv2.imwrite(os.path.join(args.output_dir, args.type+'_'+i[0:-4]+"_norm.png"), normalized_image1)
        if args.type == "train": 
            mask_img = None 
            fname = img[:-4]
            for m in masks:
                mname = m[:-9]
                if fname == mname:
                    mask_img = m
                    break
            flipped = dp.flipImage(img, DIR, False)
            cv2.imwrite(os.path.join(args.output_dir, args.type+'_'+i[0:-4]+"_1_norm.png"), flipped)
            flipped = dp.flipImage(mask_img, "..data/lung-masks")
            # print('Writing to ', os.path.join(mask_path, args.type+'_'+i[0:-4]+"_norm.png"))
            cv2.imwrite(os.path.join(mask_path, i[0:-4]+"_1_mask.png"), flipped)

            for ind, deg in enumerate([10]):
                rotated = dp.rotateImage(img, DIR, deg, False)
                cv2.imwrite(os.path.join(args.output_dir, args.type+'_'+i[0:-4]+"_"+str(ind)+"_norm.png"), rotated)
                rotated = dp.rotateImage(mask_img, deg, "..data/lung-masks")
                cv2.imwrite(os.path.join(mask_path, i[0:-4]+"_"+str(ind)+"_mask.png"), rotated)