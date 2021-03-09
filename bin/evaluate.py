#!/usr/bin/env python3
import os
import pandas as pd
from unet import UNet
import json
import numpy as np
import cv2

unet = UNet()
model = unet.model()

path = unet.args.output_dir

all_files = [i for i in os.listdir(path) if ".png" in i]

predicted_masks = []
all_masks = []
actual_masks = []

for i in all_files:
  if "mask" in i:
    if i.startswith("pred_"):
      predicted_masks.append(i)
    else:
      all_masks.append(i)

for img in predicted_masks:
  fname = img[5:-9]
  for m in all_masks:
    mname = m[0:-9]
    if fname == mname:
      actual_masks.append(m)
      break

dim = 256

predicted_masks = [cv2.imread(os.path.join(path,i))[:,:,0] for i in predicted_masks]
predicted_masks = np.array(predicted_masks).reshape((len(predicted_masks),dim,dim,1)).astype(np.float32)
actual_masks = [cv2.imread(os.path.join(path,i))[:,:,0] for i in actual_masks]
actual_masks = np.array(actual_masks).reshape((len(actual_masks),dim,dim,1)).astype(np.float32)

num_classes = 1

num_classes = predicted_masks.shape[-1]
predicted_masks = np.array([ np.argmax(predicted_masks, axis=-1)==i for i in range(num_classes) ]).transpose(1,2,3,0)

axes = (1,2) # W,H axes of each image
intersection = np.sum(np.abs(predicted_masks * actual_masks), axis=axes)
mask_sum = np.sum(np.abs(actual_masks), axis=axes) + np.sum(np.abs(predicted_masks), axis=axes)
#union = mask_sum  - intersection

smooth = 0.001
dice = 2 * (intersection + smooth)/(mask_sum + smooth)
dice = np.mean(dice)
print('Accuracy = ', dice, 'Interestion', intersection)

