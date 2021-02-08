#!/usr/bin/env python3
import sys
import os
import cv2
import pickle
import argparse
import numpy as np
from tensorflow.keras.models import load_model

def parse_args(args):
    """
    This function takes the command line arguments.        
    :param args: commandline arguments
    :return:  parsed commands
    """

    parser = argparse.ArgumentParser(description="Predicting masks")
    parser.add_argument(
                "-i",
                "--input_dir",
                default=os.getcwd(),
                help="directory where input files will be read from"
            )

    parser.add_argument(
                "-o",
                "--output_dir",
                default=os.getcwd(),
                help="directory where output files will be written to"
            )

    return parser.parse_args(args)

if __name__=="__main__":

    args = parse_args(sys.argv[1:])
    
    CURR_PATH = args.input_dir

    files = os.listdir(CURR_PATH)

    test_data = [i for i in files if ".png" in i]

    X_test = [cv2.imread(os.path.join(CURR_PATH,i))[:,:,0] for i in test_data]

    model = load_model(CURR_PATH+"/model.h5", compile=False)

    test_vol = np.array(X_test, dtype=np.float32)
    
    preds = model.predict(test_vol)

    pred_candidates = np.random.randint(1,test_vol.shape[0],len(preds))

    for i in range(len(preds)):
        img = np.squeeze(preds[pred_candidates[i]])
        cv2.imwrite(os.path.join(args.output_dir, str(test_data[i].split('.png')[0]+'_mask.png')), img)    
    


