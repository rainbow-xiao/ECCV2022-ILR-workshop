import os
import cv2
import numpy as np
import pandas as pd
import albumentations as A
import torch
from torch.utils.data import Dataset

class ImageEmbedding_Dataset(Dataset):
    def __init__(self, df, split, mode, img_size):
        if mode=='train':
            self.df = df[df['fold'] != split].reset_index(drop=True)
        elif mode=='valid':
            self.df = df[df['fold'] == split].reset_index(drop=True)
        self.img_size = img_size
        self.mean = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
        self.std  = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)
        self.transforms = self.get_transforms()

    def __len__(self):
        return self.df.shape[0]
    
    def __getitem__(self, index):
        row = self.df.iloc[index]
        image = cv2.imread(row['new_image_files'])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        images = self.transforms(image=image)['image']
        images = self.norm(images)
        images = torch.from_numpy(images.transpose((2,0,1)))
        label = row['new_labels']
        label = torch.as_tensor(label)
        return images, label
    
    def norm(self, img):
        img = img.astype(np.float32)
        img = img/255.
        img -= self.mean
        img *= np.reciprocal(self.std, dtype=np.float32)
        return img
        
    def get_transforms(self,):
        transforms=(A.Compose([
            A.Resize(self.img_size, self.img_size),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5),
            A.ShiftScaleRotate(shift_limit=0.0, scale_limit=0.2, rotate_limit=10, border_mode=0, p=0.5),
            A.CoarseDropout(max_holes=1, max_height=int(self.img_size * 0.2),max_width=int(self.img_size * 0.2), 
                            min_holes=1, min_height=int(self.img_size * 0.1),min_width=int(self.img_size * 0.1), 
                            fill_value=0, p=0.5),
        ]))
        return transforms
