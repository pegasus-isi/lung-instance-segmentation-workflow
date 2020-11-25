#!/usr/bin/env python3
import os
import cv2
import numpy as np
from cv2 import imread 

DIR = "."
X_shape = 256
norm_img = np.zeros((800,800))

class DataPreprocessing:    
    
    @staticmethod
    def normalize(i):
        
        im = cv2.resize(cv2.imread(os.path.join(DIR, i)),(X_shape,X_shape))[:,:,0]
        final_img = cv2.normalize(im,  norm_img, 0, 255, cv2.NORM_MINMAX)
        
        return final_img
    
def main():
    files = os.listdir(DIR)
    images = [i for i in files if ".png" in i]

    dp = DataPreprocessing()

    for i in images:        
        normalized_image = dp.normalize(i)
        cv2.imwrite(i.split(".png")[0]+"_norm.png", normalized_image)
        
if __name__ == "__main__":
    main()