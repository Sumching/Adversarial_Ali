import torch
import torch.nn

import torchvision.transforms as T
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import os

import numpy as np
class Read_Image(Dataset):
    def __init__(self, image_path, csv_file_dir, transforms):
        self.image_path = image_path
        self.csv_file_dir = csv_file_dir
        self.transforms = transforms

        self.all_data = pd.read_csv(csv_file_dir)

    def __getitem__(self, index):
        filename, true_label, target_label = self.all_data.iloc[index]
        image = Image.open(os.path.join(self.image_path, filename))
        image = image.convert('RGB')
        #print(np.array(image).shape)
        return self.transforms(image), true_label-1, target_label-1, filename

    def __len__(self):
        return len(self.all_data)

def get_loader(image_dir='./data/images', csv_file_dir='./data/dev.csv', mode='test',batch_size=1):
    """Build and return a data loader."""
    transform = []
    if mode == 'train':
        transform.append(T.RandomHorizontalFlip())
    #transform.append(T.CenterCrop(299))
    transform.append(T.Resize(299))
    transform.append(T.ToTensor())
    #transform.append(T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)))
    transform.append(T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
    transform = T.Compose(transform)


    dataset = Read_Image(image_dir, csv_file_dir, transform)


    data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=False,
                                  drop_last=False,
                                  num_workers=1)
    return data_loader

