#!/usr/bin/env python3
import os
import pandas as pd
from unet import UNet
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
import json
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import signal
import shutil
import tensorflow as tf
from keras import backend as keras
from reportlab.pdfgen.canvas import Canvas
from reportlab.lib.units import inch, cm
# from tensorflow.keras.callbacks import Callback
import numpy as np
import cv2
import segmentation_models as sm
from segmentation_models import get_preprocessing
from segmentation_models.metrics import iou_score
from train_analysis import GeneratePDF

unet = UNet()
config = {}
with open(os.path.join(unet.args.output_dir, 'study_results.txt'), 'r') as f:
	config = eval(f.read())['params']

# define signal handler 
def SIGTERM_handler(signum, frame):
    print("got SIGTERM")
    os.replace("model_tmp.h5", "model.h5")

signal.signal(signal.SIGTERM, SIGTERM_handler)

# model = unet.model()
BACKBONE = 'seresnet34'
preprocess_input = get_preprocessing(BACKBONE)
model = sm.Unet(BACKBONE, input_shape=(256,256,3), 
encoder_weights='imagenet', decoder_block_type='transpose')

w_path = os.path.join(unet.args.output_dir, "model_tmp.h5")
path = os.path.join(unet.args.output_dir, "model.h5")
model_copy = os.path.join(unet.args.output_dir, "model_copy.h5")

checkpoint_callback = ModelCheckpoint(w_path, monitor='loss', mode="min", save_best_only=True)
callbacks = [checkpoint_callback] 

# Compile the U-Net model
model.compile(optimizer=Adam(lr=config['lr']), loss=unet.dice_coef_loss, metrics = [iou_score, 'accuracy'])

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

# model.save(w_path)

pdf_path = os.path.join(unet.args.output_dir, 'Analysis.pdf')

#generate analysis results
pdf = GeneratePDF()
pdf.create(unet, pdf_path, history)   
os.replace(w_path, path)
shutil.copyfile(path, model_copy)
