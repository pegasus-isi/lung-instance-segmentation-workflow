#!/usr/bin/env python3

import cv2
import os, sys
import ray
#import joblib
import argparse
from ray import tune
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
        

def get_best_params(best, filename):
    """
    Saves best parameters of Optuna Study.
    """
    
    parameters = {}
    parameters["trial_id"] = best.number
    parameters["value"] = best.value
    parameters["params"] = best.params
    f = open(filename,"w")
    f.write(str(parameters))
    f.close()

def tune_unet(trial, direction="minimize"):
    """
    This function is used to train the model and call the Ray Tune callback function after every epoch for 
    hyperparameter optimization.
    :parameter config: hyperarameters list                
    """
    # model = unet.model()

    lr = trial.suggest_categorical("lr", [1e-7, 1e-8, 1e-9, 3e-5])
    # lr= trial.suggest_uniform("lr", 0.002, 0.2)
    model = sm.Unet('resnet34', input_shape=(256,256,3), encoder_weights='imagenet')

    early_stopping = EarlyStopping( monitor='val_loss', min_delta=0, patience=0)
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
            # callbacks=callbacks,
            validation_data =(valid_vol, valid_seg))
    loss = history.history["loss"]

    # os.replace(
    #     os.path.join(unet.args.output_dir, "study_checkpoint.pkl"), 
    #     os.path.join(unet.args.output_dir, "study_checkpoint_tmp.pkl")
    # )

    return sum(loss)/len(loss)


def hpo_monitor(study, trial):
    """
    Save optuna hpo study
    """
    joblib.dump(study, unet.args.output_dir+"study_checkpoint_tmp.pkl")


def create_study(abs_folder_path, write_path, result_path):
    """
    This function creates study object which contains data from each epoch, with different hyperparameters, of the training.
    :parameter checkpoint_file: File
                            Checkpoint file contains previous study object (if any) or an empty file where study object 
                            is dumped
    """
    # This seeds the hyperparameter sampling.
    np.random.seed(5)

    try:
        STUDY = joblib.load(abs_folder_path)
        todo_trials = unet.N_TRIALS - len(STUDY)
        if todo_trials > 0 :
            print("There are {} trials to do out of {}".format(todo_trials, unet.N_TRIALS))
            STUDY.optimize(tune_unet, n_trials=todo_trials,  callbacks=[hpo_monitor])
            best_trial = STUDY.best_trial
            get_best_params(best_trial, result_path)
        else:
            print("This study is finished. Nothing to do.")
    except Exception as e:
        STUDY = optuna.create_study(direction = 'minimize', study_name='Lung Segmentation')
        STUDY.optimize(tune_unet, n_trials= unet.N_TRIALS,  callbacks=[hpo_monitor])
        best_trial = STUDY.best_trial
        get_best_params(best_trial, result_path)

if __name__=="__main__":
    global unet
    unet = UNet()

    hpo_checkpoint = os.path.join(unet.args.output_dir, "study_checkpoint.pkl")
    hpo_checkpoint_tmp = os.path.join(unet.args.output_dir, "study_checkpoint_tmp.pkl")
    hpo_results = os.path.join(unet.args.output_dir, "study_results.txt")
    
    create_study(hpo_checkpoint, hpo_checkpoint_tmp, hpo_results)
    try:
        os.replace(hpo_checkpoint, hpo_checkpoint_tmp)
    except FileNotFoundError:
        pass
