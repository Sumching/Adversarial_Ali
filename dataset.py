import torch
import torch.nn

import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import os

class Read_Image(Dataset):
    def __init__(self, image_path, csv_file_dir, transforms):
        self.image_path = image_path
        self.csv_file_dir = csv_file_dir
        self.transforms = transforms

        self.all_data = pd.read_csv(csv_file_dir)

    def __getitem__(self, index):
        filename, true_label, target_label = self.all_data.iloc[index]
        image = Image.open(os.path.join(self.image_path + filename))
        return self.transforms(image), torch.FloatTensor(true_label), torch.FloatTensor(target_label)

    def __len__(self):
        return len(self.all_data)

def get_loader(image_dir='./data/images', csv_file_dir='./data/dev.csv',  batch_size=16):
    """Build and return a data loader."""
    transform = []
    if mode == 'train':
        transform.append(T.RandomHorizontalFlip())
    transform.append(T.CenterCrop(299))
    transform.append(T.Resize(299))
    transform.append(T.ToTensor())
    transform.append(T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)))
    transform = T.Compose(transform)


    dataset = Read_Image(image_dir, csv_file_dir, transform)


    data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  drop_last=False,
                                  num_workers=1)
    return data_loader

