#!/usr/bin/env python3
import os
import pandas as pd
from unet import UNet
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam
import json
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import signal
import shutil
import tensorflow as tf
from keras import backend as keras

unet = UNet()
config = {}
with open(os.path.join(unet.args.output_dir, 'study_results.txt'), 'r') as f:
	config = json.load(f)

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

# define signal handler 
def SIGTERM_handler(signum, frame):
    print("got SIGTERM")
    os.replace("model_tmp.h5", "model.h5")

signal.signal(signal.SIGTERM, SIGTERM_handler)

model = unet.model()

w_path = os.path.join(unet.args.output_dir, "model_tmp.h5")
path = os.path.join(unet.args.output_dir, "model.h5")
model_copy = os.path.join(unet.args.output_dir, "model_copy.h5")
checkpoint_callback = ModelCheckpoint(w_path, monitor='loss', save_best_only=True, save_weights_only=False)
# Enable Tune to make intermediate decisions by using a Tune Callback hook. This is Keras specific.
callbacks = [checkpoint_callback] 

# Compile the U-Net model
model.compile(optimizer=Adam(lr=config['lr']), loss=[dice_coef_loss], metrics = [dice_coef, 'binary_accuracy'])

# Call DataLoader function to get train and validation dataset
train_vol, train_seg, valid_vol, valid_seg = unet.DataLoader()

# Train the U-Net model
history = model.fit(x = train_vol, y = train_seg, batch_size = unet.args.batch_size, epochs = unet.args.epochs, validation_data =(valid_vol, valid_seg), callbacks = callbacks)


# pdf_w_path = os.path.join(unet.args.output_dir, 'Analysis_tmp.pdf')
pdf_path = os.path.join(unet.args.output_dir, 'Analysis.pdf')
#Store plots in pdf
with PdfPages(pdf_path) as pdf:
  firstPage = plt.figure(figsize=(unet.args.fig_sizex, unet.args.fig_sizey))
  text = "Model Analysis"
  firstPage.text(0.5, 0.5, text, size=24, ha="center")
  pdf.savefig()

  #summarize history for binary accuracy
  plt.figure(figsize = (unet.args.fig_sizex, unet.args.fig_sizey))
  plt.subplot(unet.args.subplotx, unet.args.subploty, 1)
  plt.plot(history.history['binary_accuracy'])
  plt.plot(history.history['val_binary_accuracy'])
  plt.title('Binary Accuracy')
#  plt.ylabel('accuracy')
#  plt.xlabel('epoch')
  plt.legend(['train', 'val'], loc='upper left')
#  pdf.savefig()

  # summarize history for loss
  plt.subplot(unet.args.subplotx, unet.args.subploty, 2)
#  plt.figure(figsize = (unet.args.fig_sizex, unet.args.fig_sizey))
  plt.plot(history.history['loss'])
  plt.plot(history.history['val_loss'])
  plt.title('Loss')
#  plt.ylabel('loss')
#  plt.xlabel('epoch')
  plt.legend(['train', 'val'], loc='upper left')
#  pdf.savefig()

  #summarize dice coefficient
  plt.subplot(unet.args.subplotx, unet.args.subploty, 3)
#  plt.figure(figsize = (unet.args.fig_sizex, unet.args.fig_sizey))
  plt.plot(history.history['dice_coef'])
  plt.plot(history.history['val_dice_coef'])
  plt.title('Dice coefficient')
#  plt.ylabel('dice coefficient')
#  plt.xlabel('epoch')
  plt.legend(['train', 'val'], loc='upper left')
  pdf.savefig()
   
  os.replace(w_path, path)
  shutil.copyfile(path, model_copy)
  # os.replace(pdf_w_path, pdf_path)
