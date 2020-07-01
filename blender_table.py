#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 16:19:24 2020

@author: abel
"""


import os
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt
import time
import cv2 

from skimage import io

# Root directory of the project
ROOT_DIR = "/home/ashfaquekp/Mask_RCNN"
DATA_DIR="/home/ashfaquekp/output_table"
# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
# Import COCO config
sys.path.append(os.path.join(ROOT_DIR, "samples/coco/"))  # To find local version
import coco



# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "coco_weights/mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

# Directory of images to run detection on

MASK_DIR="/home/ashfaquekp/output_table/mrcnn_mask/"
BASE_DIR="/home/ashfaquekp/output_table/"
class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()

# Create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# Load weights trained on MS-COCO
model.load_weights(COCO_MODEL_PATH, by_name=True)

class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']


file=open(BASE_DIR+'rgb.txt')
lines = file.read().split("\n")
print("Number of lines in associate",len(lines))
for i in range(len(lines)-1):
    image = cv2.imread(BASE_DIR+ lines[i].split(" ")[1])
    img_no=lines[i].split(" ")[1].split("_")[1][:4]
    file_img=open(MASK_DIR+str(img_no)+'.txt','w')
    
    # Run detection
    
    results = model.detect([image], verbose=0)
    
    
 
    
    r = results[0]
    masks=r['masks']
    no_masks=masks.shape[2]
    class_ids=r['class_ids']
    for i in range(no_masks):
        mask_img=r['masks'][:,:,i].astype('float')
        mask_img=mask_img*255
        mask_img=mask_img.astype('uint8')
        print("Class of ",i,":",class_names[class_ids[i]])
        mask_file_name=MASK_DIR+str(img_no)+'_'+str(+class_ids[i])+'.jpg'
        io.imsave(mask_file_name,mask_img)
        file_img.write("%s %s %s \n"%(mask_file_name,class_ids[i],class_names[class_ids[i]]))
    file_img.close()
    break
file.close()