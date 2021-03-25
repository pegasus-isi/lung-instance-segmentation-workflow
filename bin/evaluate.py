#!/usr/bin/env python3
import os
import pandas as pd
from unet import UNet
import json
import numpy as np
import cv2
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
from reportlab.pdfgen.canvas import Canvas
from reportlab.lib.units import inch, cm

unet = UNet()
model = unet.model()

path = unet.args.output_dir

all_files = [i for i in os.listdir(path) if ".png" in i]

predicted_mask_images = []
all_masks = []
actual_mask_images = []

orig_images = []
c = 3
for i in all_files:
  if "mask" in i:
    if i.startswith("pred_"):
      predicted_mask_images.append(i)
    else:
      all_masks.append(i)
  elif c > 0:
    orig_images.append(i)
    c = c-1

for img in predicted_mask_images:
  fname = img[5:-14]
  for m in all_masks:
    mname = m[0:-9]
    if fname == mname:
      actual_mask_images.append(m)
      break

dim = 256

predicted_masks = [cv2.imread(os.path.join(path,i))[:,:,0] for i in predicted_mask_images]
predicted_masks = np.array(predicted_masks).reshape((len(predicted_masks),dim,dim,1)).astype(np.float32)
actual_masks = [cv2.resize(cv2.imread(os.path.join(path,i)),(dim, dim))[:,:,0] for i in actual_mask_images]
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


canvas = Canvas("EvaluationAnalysis.pdf")
canvas.setFont("Times-Roman", 20)
canvas.drawString(inch*3,inch*11, "Evaluation Analysis")
canvas.setFont("Times-Roman", 12)
canvas.drawString(inch*3,inch*10, "Dice coefficient ="+ str(dice))
canvas.drawString(inch*3,inch*9, "Intersection = "+str(intersection))

y_inc = inch*1.25
x_inc = inch*2.5
x = inch*0.8
y = inch*.25
orig_y = y

canvas.drawString(x-inch*0.1, inch*5, "Raw Images")
for i in orig_images[0:3]:
  canvas.drawImage(i, x, y, inch, inch)
  y = y+y_inc
  print(x, y)

y = orig_y
x = x+x_inc
canvas.drawString(x-inch*0.1, inch*5, "Actual Masks")
for i in actual_mask_images[0:3]:
  canvas.drawImage(i, x, y, inch, inch)
  y = y+y_inc

y = orig_y
x = x+x_inc
canvas.drawString(x-inch*0.1, inch*5, "Predicted Masks")
for i in predicted_mask_images[0:3]:
  canvas.drawImage(i, x, y, inch, inch)
  y = y+y_inc

canvas.save()
