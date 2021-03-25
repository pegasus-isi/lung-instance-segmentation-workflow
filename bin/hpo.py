#!/usr/bin/env python3

import cv2
import os, sys
import ray
import joblib
import argparse
from ray import tune
from unet import UNet
from keras import backend as keras
import json
import numpy as np
import pandas as pd
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.optimizers import Adam
import signal, traceback
import pickle

class TuneReporterCallback(Callback):
    """
    Tune Callback for Keras. This callback is invoked every epoch.
    """
    def __init__(self, logs={}):
        self.iteration = 0
        super(TuneReporterCallback, self).__init__()

    def on_epoch_end(self, batch, logs={}):
        self.iteration += 1
        tune.report(keras_info=logs, mean_accuracy=logs.get("accuracy"), mean_loss=logs.get("loss"))

def tune_unet(config, checkpoint_dir=None):
    """
    This function is used to train the model and call the Ray Tune callback function after every epoch for 
    hyperparameter optimization.
    :parameter config: hyperarameters list                
    """

    model = unet.model()
    callbacks = [TuneReporterCallback()]

    # Compile the U-Net model
    model.compile(optimizer=Adam(lr=config["lr"]), loss=[unet.dice_coef_loss], metrics = [unet.dice_coef, 'binary_accuracy'])

    # Call DataLoader function to get train and validation dataset
    train_vol, train_seg, valid_vol, valid_seg = unet.DataLoader()

    # Train the U-Net model
    history = model.fit(x = train_vol, y = train_seg, batch_size = unet.args.batch_size, epochs = unet.args.epochs, validation_data =(valid_vol, valid_seg), callbacks = callbacks)
    with open(os.path.join(unet.args.output_dir, "study_checkpoint_tmp.pkl"), 'a+') as f:
        pickle.dump(history, f)
        print('Wrote ', history)

def create_study(checkpoint_file, hpo_results, checkpoint_tmp):
    """
    This function creates study object which contains data from each epoch, with different hyperparameters, of the training.
    :parameter checkpoint_file: File
                            Checkpoint file conatins previous study object (if any) or an empty file where study object 
                            is dumped
    """

    # This seeds the hyperparameter sampling.
    np.random.seed(5)
    hyperparameter_space = {
        "lr": tune.loguniform(0.0002, 0.2)
    }

    # Restart Ray defensively in case the ray connection is lost.
    ray.shutdown()
    ray.init(log_to_driver=False)

    path = os.path.join(unet.args.output_dir, checkpoint_file)
    write_path = os.path.join(unet.args.output_dir, checkpoint_tmp)
    result_path = os.path.join(unet.args.output_dir, hpo_results)

    if not os.path.isfile(path):
        df = pd.DataFrame(list())
        df.to_pickle(path)
    
    STUDY = joblib.load(path)

    todo_trials = unet.N_TRIALS - len(STUDY)
    analysis = tune.run(
                tune_unet,
                verbose=2,
                resources_per_trial={"gpu": 1},
                local_dir=unet.args.output_dir,
                config=hyperparameter_space,
                num_samples=todo_trials)
    print('Done checking')
    df = analysis.results_df
    df.to_pickle(write_path)
    with open(result_path, 'w') as res:
        df = analysis.get_best_config(metric="mean_loss", mode='min')
        res.write(json.dumps(df))
    #pickle.dump(df,f)


if __name__=="__main__":
    global unet
    unet = UNet()
    
    hpo_checkpoint = "study_checkpoint.pkl"
    hpo_checkpoint_tmp = "study_checkpoint_tmp.pkl"
    hpo_results = "study_results.txt"
    hpo_results_tmp = "study_results_tmp.txt"

    create_study(hpo_checkpoint, hpo_results_tmp, hpo_checkpoint_tmp)
    os.replace("study_checkpoint_tmp.pkl", "study_checkpoint.pkl")
    os.replace("study_results_tmp.txt", "study_results.txt")
