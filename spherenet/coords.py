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
import sys

# Misc
from functools import lru_cache
import cv2


def uv2xyz(uv):
    sin_u = np.sin(uv[..., 0])
    cos_u = np.cos(uv[..., 0])
    sin_v = np.sin(uv[..., 1])
    cos_v = np.cos(uv[..., 1])
    return np.stack([
        cos_v * cos_u,
        cos_v * sin_u,
        sin_v
    ], axis=-1)


def xyz2uv(xyz):
    c = np.sqrt((xyz[..., :2] ** 2).sum(-1))
    u = np.arctan2(xyz[..., 1], xyz[..., 0])
    v = np.arctan2(xyz[..., 2], c)
    return np.stack([u, v], axis=-1)


def uv2img_idx(uv, h, w, u_fov, v_fov, v_c):
    assert 0 < u_fov and u_fov < np.pi
    assert 0 < v_fov and v_fov < np.pi
    assert -np.pi < v_c and v_c < np.pi

    xyz = uv2xyz(uv.astype(np.float64))
    Ry = np.array([
        [np.cos(v_c), 0, -np.sin(v_c)],
        [0, 1, 0],
        [np.sin(v_c), 0, np.cos(v_c)],
    ])
    xyz_rot = xyz.copy()
    xyz_rot[..., 0] = np.cos(v_c) * xyz[..., 0] - np.sin(v_c) * xyz[..., 2]
    xyz_rot[..., 1] = xyz[..., 1]
    xyz_rot[..., 2] = np.sin(v_c) * xyz[..., 0] + np.cos(v_c) * xyz[..., 2]
    uv_rot = xyz2uv(xyz_rot)

    u = uv_rot[..., 0]
    v = uv_rot[..., 1]
    
    x = np.tan(u)
    y = np.tan(v) / np.cos(u)
    x = x * w / (2 * np.tan(u_fov / 2)) + w / 2
    y = y * h / (2 * np.tan(v_fov / 2)) + h / 2

    print(u,v)
    print(x,y)

    invalid = (u < -u_fov / 2) | (u > u_fov / 2) |\
              (v < -v_fov / 2) | (v > v_fov / 2)
    x[invalid] = -100
    y[invalid] = -100

    '''if (u[1] < -u_fov / 2):
        x = 0
    if (u[] > u_fov / 2):
        x = 256
    if (v[] < -v_fov / 2):
        y = 0
    if (v > v_fov / 2):
        y = 256
    '''
    return np.stack([y, x], axis=0)

import shutil
import os

path = "/home/manuel/cv/trab-cv-final/animals/train"
dirs = os.listdir( path )

file = 'image_1.txt'

# This would print all the files and directories
#for file in dirs:
#    if file.endswith('txt'):
with open(f'{path}/{file}') as f:
    print(file)
    labels, x_min, y_min, x_max, y_max = f.read().split()

    u = np.array([int(float(x_min)*256), int(float(x_max)*256)])
    v = np.array([int(float(y_min)*256), int(float(y_max)*256)])

    print(u,v)
    fov = 120 * np.pi / 180    
    #u, v = np.meshgrid(u, v)
    u = (u) * fov / 256 - fov/2
    v = (v) * fov / 256 - fov/2

    uv = np.stack([u, v], axis=-1)
    print(uv)

    img_idx = uv2img_idx(uv, 256, 256, fov, fov,0)
    #x = map_coordinates(img, img_idx, order=1)
    print(img_idx)
    #sys.exit(0)
