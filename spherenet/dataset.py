# Mathematical
import numpy as np
from scipy.ndimage.interpolation import map_coordinates
from PIL import Image
# Pytorch
#import pandas as pd

import glob

from xml.etree import ElementTree as et
import torch
from torch.utils import data
from torchvision import datasets

# Misc
from functools import lru_cache
import cv2
from matplotlib import pyplot as plt 



#imagefolder class pytoch
#https://debuggercafe.com/pytorch-imagefolder-for-training-cnn-models/



class OmniDataset(data.Dataset):
    def __init__(self, dataset, outshape=(256, 256),
                 img_mean=None, img_std=None):
        '''
        Convert classification dataset to omnidirectional version
        @dataset  dataset with same interface as torch.utils.data.Dataset
                  yield (PIL image, label) if indexing
        '''
        self.dataset = dataset
        self.outshape = outshape
        self.img_mean = img_mean
        self.img_std = img_std

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img = np.array(self.dataset[idx][0], np.float32)

        h, w = img.shape[:2]
            
        if self.img_mean is not None:
            x = x - self.img_mean
        if self.img_std is not None:
            x = x / self.img_std

        return torch.FloatTensor(x.copy()), self.dataset[idx][1]

class CustomDataset(data.Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        #self.landmarks_frame = pd.read_csv(csv_file)
        #self.idx = idx
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(glob.glob(f'{self.root_dir}/*.jpg'))

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = f'{self.root_dir}/img_{idx}.jpg'
        #image = io.imread(img_name)
        image = Image.open(img_name).convert('L')
        annot_filename = f'{self.root_dir}/img_{idx}.txt'

        return image, target

class OmniCustom(OmniDataset):
    def __init__(self, root = '/home/msnuel/trab-final-cv/animals/train', train=True,
                 download=True, *args, **kwargs):
        
        self.custom = CustomDataset(root_dir = root)
        super(OmniCustom, self).__init__(self.custom, *args, **kwargs)


if __name__ == '__main__':

    import os
    import argparse
    from PIL import Image

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--idx', nargs='+', required=True,
                        help='image indices to demo')
    parser.add_argument('--out_dir', default='datas/demo',
                        help='directory to output demo image')
    parser.add_argument('--dataset', default='OmniCustom',
                        choices=['OmniMNIST', 'OmniFashionMNIST', 'OmniCustom'],
                        help='which dataset to use')

    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    if args.dataset == 'OmniCustom':
        dataset = OmniCustom()
    for idx in args.idx:
        idx = int(idx)
        path = os.path.join(args.out_dir, '%d.jpg' % idx)
        x, label = dataset[idx]

        #print(x.shape)

        #print((x < -100).nonzero().flatten())
        Image.fromarray(x.numpy().astype(np.uint8)).save(path)
