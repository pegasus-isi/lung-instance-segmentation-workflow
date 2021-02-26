import cv2
import os, sys
import ray
import joblib
import argparse
from ray import tune
from unet import UNet
from keras import backend as keras
import pickle
import numpy as np
import pandas as pd
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.optimizers import Adam

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

    #unet = UNet()
    model = unet.model()
    
    # Enable Tune to make intermediate decisions by using a Tune Callback hook. This is Keras specific.
    callbacks = [TuneReporterCallback()] 
   
    # Compile the U-Net model
    model.compile(optimizer=Adam(lr=config["lr"]), loss=[unet.dice_coef_loss], metrics = [unet.dice_coef, 'binary_accuracy'])

    # Call DataLoader function to get train and validation dataset
    train_vol, train_seg, valid_vol, valid_seg = unet.DataLoader()

    # Train the U-Net model
    history = model.fit(x = train_vol, y = train_seg, batch_size = unet.args.batch_size, epochs = unet.args.epochs, validation_data =(valid_vol, valid_seg), callbacks = callbacks)
    

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
    
    path = os.path.join(unet.args.output_dir, checkpoint_file)
    if not os.path.isfile(path):
        df = pd.DataFrame(list())
        df.to_pickle(path)

    STUDY = joblib.load(path)
	
    todo_trials = unet.N_TRIALS - len(STUDY)
    analysis = tune.run(
                tune_unet, 
                verbose=1,
		local_dir=unet.args.output_dir,
                config=hyperparameter_space,
                num_samples=todo_trials)
    df = analysis.get_best_config(metric="mean_loss", mode='min')
    f = open(path, 'wb')
    pickle.dump(df,f) 
    
if __name__=="__main__":
    global unet
    unet = UNet()
    
    hpo_checkpoint_file = "study_checkpoint.pkl"

    model = unet.model()

    create_study(hpo_checkpoint_file)
    
#     plt.plot(config["config.lr"], config["keras_info.val_binary_accuracy"])
#     plt.xlabel('Learning rate')
#     plt.ylabel('Val Accuracy')
#     plt.show()
