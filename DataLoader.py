import os 

from PIL import Image, ImageFile
import numpy as np 
import pandas as pd 
from tqdm import tqdm
import torch 
from torchvision import datasets, transforms



class Fashion2020dataset(torch.utils.data.Dataset):
    def __init__(self, data_root, transforms, df_csv:str ):  # _to load and preprocess for the dataset. 
        
        super().__init__()
        self.data_root = data_root         
        self.transforms = transforms 
        self.imgs = list(sorted(os.listdir(os.path.join(data_root, "train"))))
        
        # _Start: read .csv with pandas for DataFormat description 
        self.df_csv = pd.read_csv(os.path.join(data_root, df_csv))  
        self.image_ids = self.df_csv["ImageId"].unique() # to get all image names
        
        
         
    def __getitem__(self, idx): # _to get a specific item.                        
        
        imgID = self.image_ids[idx]       
        #imgID = self.imgs[idx]       
        
        print(f"Image loading: {imgID}")
        
        pil_im = Image.open("{0}/train/{1}.jpg".format(self.data_root, str(imgID)))
        img = np.asarray(pil_im)
        
        images_meta = {} # 
        images_meta.update({ "image":img,
                             "shape":img.shape, 
                             "encoded_pixels": self.df_csv[self.df_csv['ImageId'] == imgID]['EncodedPixels'],
                             "class_ids" : self.df_csv[self.df_csv['ImageId'] == imgID]['ClassId']                                   
                             })
            
        # _Start: create masks with decoding and bbox from them 
        masks = []     
        boxes = [] 

        shape = images_meta.get("shape")  # _get via key of dict() 
        encoded_pixels = list(images_meta.get("encoded_pixels"))
        class_ids = list(images_meta.get("class_ids"))
        print(class_ids)
            
        # _Initialze an empty array with the same shape as the image 
        height, width = shape[:2] 
        mask = np.zeros((height, width)).reshape(-1)
        # (-1) 'The new shape should be compatible with the original shape'
            
        pbarLoad = tqdm(zip(encoded_pixels, class_ids))
        for segment, (pixel_str, class_id) in enumerate(pbarLoad):
            pbarLoad.set_description(f"Loading encoded pixels...: {segment}" )
            splitted_pixels = list(map(int, pixel_str.split())) #split the pixels string
            pixel_starts = splitted_pixels[::2] #choose every second element
            run_lengths = splitted_pixels[1::2]  #start from 1 with step size 2
               
            assert max(pixel_starts) < mask.shape[0]  
            
            pbarDecode = tqdm(zip(pixel_starts, run_lengths))    
            for pixel_start, run_length in pbarDecode:
                pbarDecode.set_description(f"Decoding masks...: {pixel_start}" )
                pixel_start = int(pixel_start) - 1
                run_length = int(run_length)
                mask[pixel_start:pixel_start+run_length] = 255 - class_id *4
                         
            
            mask = mask.reshape((height, width), order = 'F')
            masks.append(mask)
            
            # _Start: get bounding box coordinates from each mask 
            pos = np.where(mask)
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])
            # _End: get bounding box coordinates from each mask 
            
            mask = np.zeros((height, width)).reshape(-1) # re-initialize 
        # _End: create masks with decoding 
        
        
        
        # _Start: convert everything into a torch.Tensor 
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        class_ids = torch.as_tensor(class_ids, dtype=torch.uint8)
        masks = torch.as_tensor(masks, dtype=torch.uint8)  
        image_id = torch.tensor([idx])
        
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])      
        
        iscrowd = torch.zeros(len(class_ids,), dtype=torch.int64) # suppose all instances are not crowd
        # _End: convert everything into a torch.Tensor
        
        
        target = {}
        target["boxes"] = boxes
        target["class_ids"] = class_ids
        target["masks"] = masks
        target["image_id"] = image_id 
        target["area"] = area
        target["iscrowd"] = iscrowd           
        
        
        if self.transforms is not None: 
            img, target = self.transforms(img, target)           
        
        return img, target
    
    def __len__(self):          # _to return the length of data samples in the dataset. 
        return len(self.imgs)