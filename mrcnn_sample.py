#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 09:48:15 2020

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

from skimage import io

# Root directory of the project
ROOT_DIR = "/home/ashfaquekp/Mask_RCNN"

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

MASK_DIR="/home/ashfaquekp/val/0/10/mrcnn_mask/"

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


file2=open("associate.txt")
data = file2.read()
lines = data.split("\n") 

index=0

for line in lines: 
                                               #This is used to loop all images
    contents=line.split(" ")
    try:
        rgb_file=contents[0]
    except:
        print("Associate File read error at i =",index)
        continue
    print("Generating mask of Image %d"%(index))
    image = skimage.io.imread(rgb_file)
    
    # Run detection
    results = model.detect([image], verbose=0)
    
    IMAGE_NAME=rgb_file.split(".")[0].split("/")[-1]
    # Visualize results
    r = results[0]
    
    masks=r['masks']
    class_ids=r['class_ids']
    no_masks=masks.shape[2]
    
    file = open(MASK_DIR+IMAGE_NAME+'.txt',"w")
    
    for i in range(no_masks):
        mask_img=r['masks'][:,:,i].astype('float')
        mask_img=mask_img*255
        mask_img=mask_img.astype('uint8')
        
        mask_file_name=MASK_DIR+IMAGE_NAME+'_'+str(class_ids[i])+'.jpg'
        io.imsave(mask_file_name,mask_img)
        
        file.write("%s %s %s\n"%(mask_file_name,str(class_ids[i]),class_names[class_ids[i]]))
        file_class=open(MASK_DIR+'class_'+str(class_ids[i])+'.txt','a')
        file_class.write("%s \n"%(rgb_file))
        file_class.close()
    file.close()
index+=1


    
file2.close()    



#visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], 
#                            class_names, r['scores'])