import os
from pathlib import Path
from shutil import copyfile

img_dir = Path(os.path.join("."), "../data/lung-images")

train_dir = Path(os.path.join("."), "../data/train")
test_dir = Path(os.path.join("."), "../data/test")
val_dir = Path(os.path.join("."), "../data/val")

train= []
test = []
val = []

i = 0
l = 801
for f in img_dir.iterdir():
    if f.name.endswith(".png"):
        i += 1
        # if f.name in IGNORE_IMAGES: continue
        if i+1 <= 0.7*l:
            copyfile(os.path.join(img_dir, f.name), os.path.join(train_dir, f.name))
        elif i+1 <= 0.9*l:
            copyfile(os.path.join(img_dir, f.name), os.path.join(val_dir, f.name))
        else:
            copyfile(os.path.join(img_dir, f.name), os.path.join(test_dir, f.name))