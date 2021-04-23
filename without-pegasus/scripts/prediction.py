#!/usr/bin/env python3
import sys
import os
import cv2
import pickle
import argparse
import numpy as np
from tensorflow.keras.models import load_model
import tensorflow as tf
from unet import UNet

print('##########Version = ', tf.__version__)


def save_result(save_path, npyfile, test_files):
    for i, item in enumerate(npyfile):
        result_file = test_files[i]
        print('Shape ',item.shape)
        img = (item[:, :, 0] * 255.).astype(np.float32)

        filename, fileext = os.path.splitext(os.path.basename(result_file))

        result_file = os.path.join(save_path, "%s_predict%s" % (filename, fileext))
        print('Writing to ', result_file)
        cv2.imwrite(result_file, img)

if __name__=="__main__":
    unet = UNet()
    CURR_PATH = os.path.join(".", "data/test_processed")
    
    files = os.listdir(CURR_PATH)

    test_data = [i for i in files if ".png" in i and i.startswith("test_")]
	
    dim = 256
    actual_images = [i for i in test_data]
    X_test = [cv2.imread(os.path.join(CURR_PATH,i), 1) for i in actual_images]

    # actual_images = [os.path.join("./data/test_processed/", "test_CHNCXR_0003_0_norm.png")]
    # X_test = [cv2.imread(os.path.join("./data/test_processed/", "test_CHNCXR_0003_0_norm.png"), 1)]
    

    X_test = np.array(X_test).reshape((len(X_test),dim,dim,3)).astype(np.float32)
    model = load_model(os.path.join(unet.args.output_dir,"model_copy.h5"), compile=False)
    preds = model.predict(X_test)

    for i in range(len(preds)):
        im = preds[i] #np.squeeze(preds[i])
        im[im>0.5] = 1
        im[im<0.5] = 0
        width,height=im.shape[1],im.shape[0]
        img=cv2.imread(actual_images[i],0)
        print(im.shape)
        img=cv2.resize(img,(256,256)).reshape(256,256, 1)
        masked_img=np.squeeze(img*im.reshape(256,256, 1))
        masked_img[masked_img>0.5] = 255
        masked_img[masked_img<0.5] = 0
        masked_img=cv2.resize(masked_img,(width,height)).astype(np.float32)
        # cv2.imwrite(os.path.join(unet.args.output_dir,'pred_'+ str(test_data[i].split('.png')[0][5:]+'_mask.png')), masked_img)   
        cv2.imwrite(os.path.join(unet.args.output_dir,'pred_'+ str("test_mask.png")), masked_img)   