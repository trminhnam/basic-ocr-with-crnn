import json
import os

import albumentations as A
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchvision
import yaml
from torchvision.io import read_image

from utils import get_config


class OCRDataset(torch.utils.data.Dataset):
    def __init__(self, img_dir, label_dir, size, letters, max_len=100):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.size = size
        self.letters = letters
        self.max_len = max_len
        
        with open(label_dir, 'r', encoding='utf-8') as f:
            self.labels = json.load(f)
        self.img_paths = [os.path.join(img_dir, img_name) for img_name in self.labels.keys() 
                          if os.path.exists(os.path.join(img_dir, img_name))]
        
        # print(f'Number of images: {len(self.img_paths)}')
        # print(f'Number of labels: {len(self.labels)}')
        # print(f'Example: {self.img_paths[0]} -> {self.labels[os.path.basename(self.img_paths[0])]}')
        
        self.transforms = torchvision.transforms.Compose([
            torchvision.transforms.Resize(size),
            torchvision.transforms.ToPILImage(),
            torchvision.transforms.ToTensor(),
        ])
        
        self.label_encoder = {letter: idx + 1 for idx, letter in enumerate(letters)}

    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img = read_image(img_path)
        label = self.labels[os.path.basename(img_path)]
        
        # if self.transform:
            # img = self.transform(image=img)['image']
        img = self.transforms(img)
        
        target = torch.zeros(self.max_len, dtype=torch.long)
        for i, letter in enumerate(label):
            target[i] = self.label_encoder[letter]

        return {
            "image": img,
            "label": label,
            "target": target,
            "length": len(label)
        }

if __name__ == '__main__':
    dataset = OCRDataset(
        get_config['image_dir'], 
        get_config['label_dir'], 
        (get_config['image']['height'], get_config['image']['width']), 
        get_config['letters']
    )
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)
    sample = next(iter(dataloader))
    
    img = sample['image']
    label = sample['label']
    target = sample['target']
    print(img.shape)
    print(label)
    print(target)
    
    plt.imshow(img[0].permute(1, 2, 0))
    plt.show()
