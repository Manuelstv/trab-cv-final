import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

from torch import nn 
from torchvision.datasets import ImageFolder

import torch.optim as optim

from torch.utils.data import random_split

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

data = 'trainset/'
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.1307,), (0.3081,))])
trainset = ImageFolder(data,transform=transform)
trainset.classes,len(trainset)

trainset, testset = random_split(trainset, [40000,2000])
len(trainset),len(testset)

trainloader =  torch.utils.data.Dataset()
#testloader =  torch.utils.data.Dataset(testset, batch_size=4,shuffle=False, num_workers=2)

print(trainloader)
        
        
        
        
        
