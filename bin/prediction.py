#!/usr/bin/env python3
import sys
import os
import cv2
import pickle
import argparse
import numpy as np
from tensorflow.keras.models import load_model
from unet import UNet

if __name__=="__main__":
    unet = UNet()
    CURR_PATH = unet.args.input_dir

    files = os.listdir(CURR_PATH)

    test_data = [i for i in files if ".png" in i and i.startswith("test_")]
	
    dim = 256
    X_test = [cv2.imread(os.path.join(CURR_PATH,i))[:,:,0] for i in test_data]
    X_test = np.array(X_test).reshape((len(X_test),dim,dim,1)).astype(np.float32)

    model = load_model(os.path.join(CURR_PATH,"model.h5"), compile=False)
    preds = model.predict(X_test)

    #pred_candidates = np.random.randint(1, X_test.shape[0], len(preds))

    for i in range(len(preds)):
        img = np.squeeze(preds[i])
        cv2.imwrite(os.path.join(unet.args.output_dir, str(test_data[i].split('.png')[0]+'_mask.png')), img)    
    


