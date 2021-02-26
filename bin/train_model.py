#!/usr/bin/env python3
import os
import pandas as pd
from unet import UNet
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam

unet = UNet()

config = pd.read_pickle(os.path.join(unet.args.output_dir,'study_checkpoint.pkl'))

model = unet.model()
	
checkpoint_callback = ModelCheckpoint(os.path.join(unet.args.output_dir, "model.h5"), monitor='loss', save_best_only=True, save_weights_only=False, save_freq=2)
# Enable Tune to make intermediate decisions by using a Tune Callback hook. This is Keras specific.
callbacks = [checkpoint_callback] 

# Compile the U-Net model
model.compile(optimizer=Adam(lr=config["lr"]), loss=[unet.dice_coef_loss], metrics = [unet.dice_coef, 'binary_accuracy'])

# Call DataLoader function to get train and validation dataset
train_vol, train_seg, valid_vol, valid_seg = unet.DataLoader()

# Train the U-Net model
model.fit(x = train_vol, y = train_seg, batch_size = unet.args.batch_size, epochs = unet.args.epochs, validation_data =(valid_vol, valid_seg), callbacks = callbacks)

