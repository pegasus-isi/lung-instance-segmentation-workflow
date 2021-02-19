import cv2
import os, sys
import ray
import joblib
import argparse
from ray import tune
from unet import UNet
from keras import backend as keras
import pickle

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

    # Enable Tune to make intermediate decisions by using a Tune Callback hook. This is Keras specific.
    callbacks = [TuneReporterCallback()] 

    # Compile the U-Net model
    model.compile(optimizer=Adam(lr=config["lr"]), loss=[dice_coef_loss], metrics = [dice_coef, 'binary_accuracy'])

    # Call DataLoader function to get train and validation dataset
    train_vol, train_seg, valid_vol, valid_seg = unet.DataLoader()

    # Train the U-Net model
    history = model.fit(x = train_vol, y = train_seg, batch_size = unet.BATCH_SIZE, epochs = unet.EPOCHS, validation_data =(valid_vol, valid_seg), callbacks = callbacks)


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
    # print('STUDY', STUDY)
    todo_trials = unet.N_TRIALS - len(STUDY)
    analysis = tune.run(
                tune_unet, 
                verbose=1,
                config=hyperparameter_space,
                num_samples=todo_trials)            
    df = analysis.get_best_config(metric="mean_loss", mode='min')
    f = open(path, 'wb')
    pickle.dump(df,f) 
    
if __name__=="__main__":
    global unet
    unet = UNet()
    
    hpo_checkpoint_file = "study_checkpoint.pkl"

    create_study(hpo_checkpoint_file)
    
#     plt.plot(config["config.lr"], config["keras_info.val_binary_accuracy"])
#     plt.xlabel('Learning rate')
#     plt.ylabel('Val Accuracy')
#     plt.show()