import os 
import sys 
import argparse
from _thread import *

from PIL import Image, ImageFile
import numpy as np 
import pandas as pd 
import json
from pathlib import Path 
import cv2
from tqdm import tqdm 

from DataLoader import Fashion2020dataset




# _Start: change working directory scope 
cwd = os.getcwd()
os.chdir(cwd)
# _End: change working directory scope 


# _Start: argparse 
ap = argparse.ArgumentParser()
ap.add_argument('-r', '--root', type=str, default="../imaterialist-fashion-2020-fgvc7",
                 help="path to root directory")
ap.add_argument('-c', '--df-csv', type=str, default="../imaterialist-fashion-2020-fgvc7/train.csv", 
                 help="path to train.csv")
ap.add_argument('-s', '--df-json', type=str, default="../imaterialist-fashion-2020-fgvc7/label_descriptions.json",
                 help="path to label_descriptions.json")
ap.add_argument('-t', '--train-json', type=str, default="../imaterialist-fashion-2020-fgvc7/train.json",
                 help="path to COCO formatted train.json")
ap.add_argument('-v', '--val-json', type=str, default="../imaterialist-fashion-2020-fgvc7/validation.json",
                 help="path to COCO formatted validation.json")

args = vars(ap.parse_args())
print(args["df_json"])
# _End: argparse 




# _Start: split data into train and validation in a 9:1 ratio
def split_train_validation(data_lists:list):
    img_lists = data_lists
    #print(img_files)
    train_idx = [] 
    val_idx   = [] 
    
    # _Start: split 
    for i, img_file in enumerate(img_lists): 
#    for i, img_file in enumerate(range(10)): 
        if i % 10 == 0: 
            val_idx.append(i)
        else:
            train_idx.append(i)    
    # _End: split 
    
    print(f"Number of training samples: {len(train_idx)}")
    print(f"Number of validation samples: {len(val_idx)}")
    
    return train_idx, val_idx
# _End: split data into train and validation in a 9:1 ratio


def get_contour_pts(mask_img):
    mask = mask_img

    assert mask is not None, "mask is empty..."

    ret, thresh = cv2.threshold(mask, 127, 255,  0)

    _, orig_contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)
        
    sorted_contours = [] 
    cnt_area = [] 

    for cnt in orig_contours:
        if len(cnt.shape) == 3 and cnt.shape[1] == 1:
            cnt_area.append(cv2.contourArea(cnt))
            sorted_contours.append(cnt.reshape(-1).tolist())

    contour = [p for p in sorted_contours if len(p) > 4]
    area = [cnt_area[i] for i, p in enumerate(sorted_contours) if len(p)>4]

    return (contour, area)


def generate_COCO_formatted_json(Dataloader:Fashion2020dataset, data_idx:list ,json_filepath:str):
    annotations = []
    images = []
    categories = []
    
    for i, idx in tqdm(enumerate(data_idx), desc="JSON processing..."):
        img, target = Dataloader[idx]
        
        image_filename = '{}.jpg'.format(target["image_id"])

        if len(img.shape) == 3:
            w,h, _ = img.shape
        else:
            w,h = img.shape 
        
        if image_filename not in Dataloader.img_lists:
            continue 
            
        
        for i, mask in enumerate(target["masks"]):
            
            contour, area = get_contour_pts(mask.numpy())
            
            if not contour:
                continue 
                
            xmin, ymin, xmax, ymax = target["boxes"][i] 
            annotations.append({"segmentation": contour,
                                "iscrowd": 0, 
                                "image_id": target["image_id"],
                                "bbox": [xmin, ymin, xmax-xmin, ymax-ymin],
                                "area": area[0],
                                "category_id": target["class_ids"][i],
                                "id": target["image_id"] + str(i) })
            
            categories.append({"supercategory":df_categories["supercategory"][target["class_ids"][i]] ,
                               "id":target["class_ids"][i],
                               "name": df_categories["name"][target["class_ids"][i]]    })
            
        
        images.append({"file_name" : image_filename,
                       "height": h,
                       "width" : w,
                       "id" : target["image_id"]  })
            

    with open(json_filepath, "w") as f:
        json.dump({"images": images, "annotations": annotations, "categories": categories}, f)
        
    return



if __name__ == "__main__":

    Dataloader = Fashion2020dataset(root= args["root"], transforms=None, csv_path= args["df_csv"] )

    train_idx, val_idx = split_train_validation(Dataloader.img_lists)

    label_ds_json = args["df_json"]

    print("Label df", label_ds_json)
    # _Start: get label descriptions from json 
    with open(label_ds_json, 'r') as file:
        label_desc = json.load(file)
    # _End: get label descriptions  from json 


    # _Start: Classes and Attributes processing 
    df_categories = pd.DataFrame(label_desc['categories'])
##    df_attributes = pd.DataFrame(label_desc['attributes'])
    # _End: Classes and Attributes processing 


    # _Start: subthread 
    start_new_thread(generate_COCO_formatted_json(Dataloader= Dataloader,
                             data_idx= train_idx,
                             json_filepath=args["train_json"]) ,
                             (enclosure_queue,)    )

    start_new_thread(generate_COCO_formatted_json(Dataloader= Dataloader,
                             data_idx= train_idx,
                             json_filepath=args["val_json"]) ,
                             (enclosure_queue,)    )
    # _End: subthread 

    print("End...")
