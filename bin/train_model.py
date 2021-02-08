#!/usr/bin/env python3
import os
import sys
import ray
import cv2
import joblib
import argparse
import numpy as np 
import pandas as pd
from ray import tune
from keras import backend as keras
from cv2 import imread, createCLAHE 
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.optimizers import Adam 
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, Callback
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau

def parse_args(args):
    """
        This function takes the command line arguments.        
        :param args: commandline arguments
        :return:  parsed commands
    """
    parser = argparse.ArgumentParser(description="Lung Image Segmentation Using UNet Architecture")
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
    
    parser.add_argument('-epochs',  metavar='num_epochs', type=int, default = 5, help = "Number of training epochs")
    parser.add_argument('--batch_size',  metavar='batch_size', type=int, default = 16, help = "Batch Size")

    return parser.parse_args(args)    

class UNet:
    def DataLoader(self):
        """
        This function takes all the training images and masks from the current working directory and converts
        the images into a numpy array.        
        :return:  
            X_train: ndarray
                      2D array containing train images            
            y_train: ndarray
                      2D array containing train masks  
            X_valid: ndarray
                      2D array containing validation images  
            y_valid: ndarray
                      2D array containing validation masks  
        """
        path = OUTPUT_FOLDER        
        valid_data = []
        mask_valid = []
        files = os.listdir(path)
        all_images = [i for i in files if ".png" in i]
        masks = [i for i in all_images if "mask" in i]
        images = [i.split("_mask_")[0]+"_norm.png" for i in masks]
        for i in range(len(images)-1, int(0.7*(len(images))), -1):
            valid_data.append(images[i])
        for i in valid_data:
            mask_valid.append(i.split("_norm")[0]+"_mask_norm.png")
        for i in mask_valid:
            valid_data.append(i)
        images = [i for i in images if i not in valid_data]
        masks = [i for i in masks if i not in valid_data]             

        X_train = [cv2.imread(os.path.join(path,i))[:,:,0] for i in train_data if 'mask' not in i]
        y_train = [cv2.imread(os.path.join(path,i))[:,:,0] for i in train_data if 'mask' in i]

        X_valid = [cv2.imread(os.path.join(path,i))[:,:,0] for i in valid_data if 'mask' not in i]
        y_valid = [cv2.imread(os.path.join(path,i))[:,:,0] for i in valid_data if 'mask' in i]

        dim = 256

        X_train = np.array(X_train).reshape((len(X_train),dim,dim,1))
        y_train = np.array(y_train).reshape((len(y_train),dim,dim,1))
        X_valid = np.array(X_valid).reshape((len(X_valid),dim,dim,1))
        y_valid = np.array(y_valid).reshape((len(y_valid),dim,dim,1))
        assert X_train.shape == y_train.shape
        assert X_valid.shape == y_valid.shape

        return X_train.astype(np.float32), y_train.astype(np.float32), X_valid.astype(np.float32), y_valid.astype(np.float32)
    
    def model(self, input_size=(256,256,1)):
        """
        This function is the U-Net architecture. It has 2 steps, contraction path (encoder) and symmetric expansion path (decoder).
        The encoder is a traditional stack of convolution and max pooling layers. The decoder is used to enable precise localization
        using transposed convolutions and thus is an end-to-end fully convolutional network (FCN).
        :parameter input_size: input image ndarray size
        :return:  
            Model:  U-Net model
        """

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
    """
    This function is used to gauge the similarity of two samples. It is also called F1-score.
    :parameter y_true: actual mask of the image
    :parameter y_pred: predicted mask of the image
    :return: dice_coefficient value        
    """
    y_true_f = keras.flatten(y_true)
    y_pred_f = keras.flatten(y_pred)
    intersection = keras.sum(y_true_f * y_pred_f)
    return (2. * intersection + 1) / (keras.sum(y_true_f) + keras.sum(y_pred_f) + 1)
    
def dice_coef_loss(y_true, y_pred):
    """
    This function is used to gauge the similarity of two samples. It is also called F1-score.
    :parameter y_true: actual mask of the image
    :parameter y_pred: predicted mask of the image
    :return: dice_coefficient value        
    """
    return -dice_coef(y_true, y_pred)


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

def tune_unet(config):
    """
    This function is used to train the model and call the Ray Tune callback function after every epoch for 
    hyperparameter optimization.
    :parameter config: hyperarameters list                
    """

    unet = UNet()
    model = unet.model()
    checkpoint_callback = ModelCheckpoint(os.path.join(OUTPUT_FOLDER, "model.h5"), monitor='loss', save_best_only=True, save_weights_only=False, save_freq=2)

    # Enable Tune to make intermediate decisions by using a Tune Callback hook. This is Keras specific.
    callbacks = [checkpoint_callback, TuneReporterCallback()] 

    # Compile the U-Net model
    model.compile(optimizer=Adam(lr=config["lr"]), loss=[dice_coef_loss], metrics = [dice_coef, 'binary_accuracy'])

    # Call DataLoader function to get train and validation dataset
    train_vol, train_seg, valid_vol, valid_seg = unet.DataLoader()

    # Train the U-Net model
    model.fit(x = train_vol, y = train_seg, batch_size = BATCH_SIZE, epochs = EPOCHS, validation_data =(valid_vol, valid_seg), callbacks = callbacks)

def create_study(checkpoint_file):
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
    
    if not os.path.isfile(checkpoint_file):
        df = pd.DataFrame(list())
        df.to_pickle(os.path.join(OUTPUT_FOLDER, checkpoint_file))
    
    STUDY = joblib.load("study_checkpoint.pkl")
    todo_trials = N_TRIALS - len(STUDY)
    analysis = tune.run(
                tune_unet, 
                verbose=1,
                config=hyperparameter_space,
                num_samples=todo_trials)            
    df = analysis.results_df
    df.to_pickle(os.path.join(OUTPUT_FOLDER, checkpoint_file)) 
    
if __name__=="__main__":
    global EPOCHS
    global BATCH_SIZE
    global N_TRIALS
    global CURR_PATH
    global loss_history
    global N_TRIALS
    
    args = parse_args(sys.argv[1:])
    OUTPUT_FOLDER = args.input_dir

    EPOCHS = args.epochs
    BATCH_SIZE = args.batch_size
    N_TRIALS = 1

    hpo_checkpoint_file = "study_checkpoint.pkl"

    create_study(hpo_checkpoint_file)


