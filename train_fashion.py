import random 
import os
import sys 

import numpy as np 
import cv2
import json 
from pathlib import Path
import matplotlib.pyplot as plt 

import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

from detectron2 import model_zoo
from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import DatasetCatalog,MetadataCatalog

from detectron2.data.datasets import register_coco_instances
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader

data_dir = Path("../imaterialist-fashion-2020-fgvc7")
train_json_path = os.path.join(data_dir, 'trainFix.json')
validation_json_path = os.path.join(data_dir, 'validationFix.json')

train_data_name = 'fashion_train'
val_data_name = 'fashion_val'


register_coco_instances(train_data_name, {}, train_json_path, os.path.join(data_dir, 'train') )
register_coco_instances(val_data_name, {}, validation_json_path, os.path.join(data_dir, 'train') )


train_metadata = MetadataCatalog.get(train_data_name)

dataset_dicts = DatasetCatalog.get(name=train_data_name)

#_Start: Set config 
#Initialize with default configuration
cfg = get_cfg()

# update configuration with MaskRCNN configuration
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))

# Let's replace the detectron2 default train dataset with our train dataset.
cfg.DATASETS.TRAIN = (train_data_name,)

# No metric implemented for the test dataset, we will have to update cfg.DATASET.TEST with empty tuple
cfg.DATASETS.TEST = (val_data_name,)

# data loader configuration
cfg.DATALOADER.NUM_WORKERS = 2

# Update model URL in detectron2 config file
# Let training initialize from model zoo
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  

# batch size
cfg.SOLVER.IMS_PER_BATCH = 2

# choose a good learning rate
cfg.SOLVER.BASE_LR = 0.0005

# We need to specify the number of iteration for training in detectron2, not the number of epochs.
cfg.SOLVER.MAX_ITER = 10000
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128

# number of output class
# we have only one class that is Backpack
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 46
#_End: Set config 

# update create ouptput directory 
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

# _Start: training 
# Create a trainer instance with the configuration.
trainer = DefaultTrainer(cfg) 

# if resume=False, because we don't have trained model yet. It will download model from model url and load it
trainer.resume_or_load(resume=False)

# start training
trainer.train()
# _End: training