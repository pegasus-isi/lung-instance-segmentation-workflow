#!/usr/bin/env python3
import argparse
import numpy as np 
import tensorflow as tf
import pandas as pd
import joblib
import os
import sys
from cv2 import imread, createCLAHE 
import cv2
import pickle
from glob import glob
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from keras import backend as keras
from tensorflow.keras.optimizers import Adam 
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, Callback
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau
import ray
from ray import tune

def parse_args(args):
    parser = argparse.ArgumentParser(description='Lung Image Segmentation Using UNet Architecture')
    parser.add_argument('-epochs',  metavar='num_epochs', type=int, default = 5, help = "Number of training epochs")
    parser.add_argument('--batch_size',  metavar='batch_size', type=int, default = 16, help = "Batch Size")
    #parser.add_argument('-n_trials',  metavar='num_trials', type=int, default = 3, help = "Number of Trials")
    return parser.parse_args()

class TuneReporterCallback(Callback):
    def __init__(self, logs={}):
        self.iteration = 0
        super(TuneReporterCallback, self).__init__()

    def on_epoch_end(self, batch, logs={}):
        self.iteration += 1
        tune.report(keras_info=logs, mean_accuracy=logs.get("accuracy"), mean_loss=logs.get("loss"))

class UNet:
    def DataLoader(self):
                
        infile = open(CURR_PATH+"/data_split.pkl",'rb')
        new_dict = pickle.load(infile)
        infile.close()

        path = CURR_PATH

        train_data = new_dict['train']
        valid_data = new_dict['valid']
        test_data = new_dict['test']

        X_train = [cv2.imread(os.path.join(path,i))[:,:,0] for i in train_data if 'mask' not in i]
        y_train = [cv2.imread(os.path.join(path,i))[:,:,0] for i in train_data if 'mask' in i]

        X_valid = [cv2.imread(os.path.join(path,i))[:,:,0] for i in valid_data if 'mask' not in i]
        y_valid = [cv2.imread(os.path.join(path,i))[:,:,0] for i in valid_data if 'mask' in i]

        X_test = [cv2.imread(os.path.join(path,i))[:,:,0] for i in test_data]

        dim = 256

        X_train = np.array(X_train).reshape((len(X_train),dim,dim,1))
        y_train = np.array(y_train).reshape((len(y_train),dim,dim,1))
        X_valid = np.array(X_valid).reshape((len(X_valid),dim,dim,1))
        y_valid = np.array(y_valid).reshape((len(y_valid),dim,dim,1))
        X_test = np.array(X_test).reshape((len(X_test),dim,dim,1))
        assert X_train.shape == y_train.shape
        assert X_valid.shape == y_valid.shape

        return X_train.astype(np.float32), y_train.astype(np.float32), X_valid.astype(np.float32), y_valid.astype(np.float32), X_test.astype(np.float32)
    
    def model(self, input_size=(256,256,1)):
        inputs = Input(input_size)
        conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
        conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

        conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
        conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

        conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
        conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

        conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
        conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

        conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool4)
        conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)

        up6 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv5), conv4], axis=3)
        conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(up6)
        conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv6)

        up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv6), conv3], axis=3)
        conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(up7)
        conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)

        up8 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv7), conv2], axis=3)
        conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(up8)
        conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv8)

        up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=3)
        conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(up9)
        conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv9)

        conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)

        return Model(inputs=[inputs], outputs=[conv10])

def dice_coef(y_true, y_pred):
    y_true_f = keras.flatten(y_true)
    y_pred_f = keras.flatten(y_pred)
    intersection = keras.sum(y_true_f * y_pred_f)
    return (2. * intersection + 1) / (keras.sum(y_true_f) + keras.sum(y_pred_f) + 1)
    
def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)



class TuneReporterCallback(Callback):
    def __init__(self, logs={}):
        self.iteration = 0
        super(TuneReporterCallback, self).__init__()

    def on_epoch_end(self, batch, logs={}):
        self.iteration += 1
        tune.report(keras_info=logs, mean_accuracy=logs.get("accuracy"), mean_loss=logs.get("loss"))

def tune_unet(config):
    unet = UNet()
    model = unet.model()
    checkpoint_callback = ModelCheckpoint(os.path.join(CURR_PATH, "model.h5"), monitor='loss', save_best_only=True, save_freq=2)
    callbacks = [checkpoint_callback, TuneReporterCallback()]
    model.compile(optimizer=Adam(lr=config["lr"]), loss=[dice_coef_loss], metrics = [dice_coef, 'binary_accuracy'])
    train_vol, train_seg, valid_vol, valid_seg, test_vol = unet.DataLoader()
    loss_history = model.fit(x = train_vol, y = train_seg, batch_size = BATCH_SIZE, epochs = EPOCHS, validation_data =(valid_vol, valid_seg), callbacks = callbacks)

def create_study(checkpoint_file):
    np.random.seed(5)  
    hyperparameter_space = {
        "lr": tune.loguniform(0.0002, 0.2)
        }   
    
    ray.shutdown()  
    ray.init(log_to_driver=False)
    
    STUDY = joblib.load("study_checkpoint.pkl")
    todo_trials = N_TRIALS - len(STUDY)
    analysis = tune.run(
                tune_unet, 
                verbose=1,
                config=hyperparameter_space,
                num_samples=todo_trials)            
    df = analysis.results_df
    df.to_pickle(checkpoint_file) 
        

def main():
    
    global EPOCHS
    global BATCH_SIZE
    global N_TRIALS
    global CURR_PATH
    global loss_history
    N_TRIALS = 1
    CURR_PATH = os.getcwd()
    
    args = parse_args(sys.argv[1:])

    EPOCHS = args.epochs
    BATCH_SIZE = args.batch_size
    #N_TRIALS = args.n_trials

    hpo_checkpoint_file = "study_checkpoint.pkl"

    create_study(hpo_checkpoint_file)
     

#__name__ prints tensorflow.keras.optimizers

main()


