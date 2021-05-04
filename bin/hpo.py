#!/usr/bin/env python3

import cv2
import os, sys
#import joblib
import argparse
from unet import UNet
from keras import backend as keras
import json
import numpy as np
import pandas as pd
from tensorflow.keras.callbacks import Callback, EarlyStopping
from tensorflow.keras.optimizers import Adam
import signal, traceback
import pickle, shutil
import tarfile
import segmentation_models as sm
import optuna
from segmentation_models.metrics import iou_score
import joblib
        

def get_best_params(best, filename, root):
    """
    Saves best parameters of Optuna Study.
    """
    
    parameters = {}
    parameters["trial_id"] = best.number
    parameters["value"] = best.value
    parameters["params"] = best.params
    f = open(os.path.join(root, filename),"w")
    f.write(str(parameters))
    f.close()

def tune_unet(trial, direction="minimize"):
    """
    This function is used to train the model and call the Ray Tune callback function after every epoch for 
    hyperparameter optimization.
    :parameter config: hyperarameters list                
    """

    lr = trial.suggest_categorical("lr", [1e-2, 1e-3, 1e-1, 3e-5, 0.012961042001467773])
    model = sm.Unet('resnet34', input_shape=(256,256,3), encoder_weights='imagenet')

    early_stopping = EarlyStopping( monitor='loss', min_delta=0, patience=4)
    callbacks = [early_stopping]

    # Compile the U-Net model
    model.compile(optimizer=Adam(lr=lr), loss=[unet.dice_coef_loss], metrics = [iou_score, 'accuracy'])

    # Call DataLoader function to get train and validation dataset
    train_vol, train_seg, valid_vol, valid_seg = unet.DataLoader()

    # Train the U-Net model
    history = model.fit(
            x = train_vol, 
            y = train_seg, 
            batch_size = unet.args.batch_size, 
            epochs = unet.args.epochs,
            callbacks=callbacks,
            validation_data =(valid_vol, valid_seg))
    loss = history.history["loss"]

    return sum(loss)/len(loss)


def hpo_monitor(study, trial):
    """
    Save optuna hpo study
    """
    joblib.dump(study, os.path.join(unet.args.output_dir, "study_checkpoint_tmp.pkl"))



def create_study(abs_folder_path, write_path, result_path):
    """
    This function creates study object which contains data from each epoch, with different hyperparameters, of the training.
    :parameter checkpoint_file: File
                            Checkpoint file contains previous study object (if any) or an empty file where study object 
                            is dumped
    """
    # This seeds the hyperparameter sampling.

    # if not os.path.exists(abs_folder_path):
    #     df = pd.DataFrame(list())
    #     df.to_pickle(abs_folder_path)
            

    try:
        STUDY = joblib.load(abs_folder_path)
        todo_trials = unet.N_TRIALS - len(STUDY.trials_dataframe())
        if todo_trials > 0 :
            print("There are {} trials to do out of {}".format(todo_trials, unet.N_TRIALS))
            STUDY.optimize(tune_unet, n_trials=todo_trials,  callbacks=[hpo_monitor])
            best_trial = STUDY.best_trial
            get_best_params(best_trial, result_path, unet.args.output_dir)
        else:
            print("This study is finished. Nothing to do.")
    except Exception as e:
        STUDY = optuna.create_study(direction = 'minimize', study_name='Lung Segmentation')
        STUDY.optimize(tune_unet, n_trials= unet.N_TRIALS,  callbacks=[hpo_monitor])
        best_trial = STUDY.best_trial
        get_best_params(best_trial, result_path, unet.args.output_dir)


def SIGTERM_handler(signum, frame):
    print("got SIGTERM")
    os.rename("study_checkpoint_tmp.pkl", "study_checkpoint.pkl")

signal.signal(signal.SIGTERM, SIGTERM_handler) 

if __name__=="__main__":
    global unet
    unet = UNet()
    unet.DataLoader()

    hpo_checkpoint = os.path.join(unet.args.output_dir, "study_checkpoint.pkl")
    hpo_checkpoint_tmp = os.path.join(unet.args.output_dir, "study_checkpoint_tmp.pkl")
    hpo_results = "study_results.txt"
    
    create_study(hpo_checkpoint, hpo_checkpoint_tmp, hpo_results)
    try:
        os.rename(hpo_checkpoint_tmp, hpo_checkpoint)
    except FileNotFoundError:
        pass
