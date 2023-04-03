# -*- coding: utf-8 -*-

import os
import pandas as pd
from fileinput import input as fileinput
import glob
import shutil
from sklearn.model_selection import train_test_split

DIR_PATH = '/home/msnuel/trab-final-cv/animais_sph'

dir = os.listdir(DIR_PATH)

#Utility function to copy images 
def copy_files_to_folder(list_of_files, destination_folder):
    for file in list_of_files:
            try:
                shutil.copy(f"{DIR_PATH}/{file}", f"{destination_folder}")
            except:
                print(file)
                assert False

#only slect files with both .txt and .jpg formats
annots = [x for x in os.listdir(DIR_PATH) if x[-3:] == "txt"]
images = [x.replace("jpg", "txt") for x in os.listdir(DIR_PATH) if x[-3:] == "jpg"]

def intersection(list_1, list_2):
    list_3 = [value for value in list_1 if value in list_2]
    return list_3

annots = intersection(images, annots) 
images = [x.replace("txt", "jpg") for x in annots]

n_imgs = len(images)
print("Número de imagens: {}".format(n_imgs))

images.sort()
annots.sort()
k = 3

PATH = '/home/msnuel/trab-final-cv'

for n in range(0, k):
    img_fold = images[int(n_imgs*n/k):int(n_imgs*(n+1)/k)]
    annots_fold = annots[int(n_imgs*n/k):int(n_imgs*(n+1)/k)]
    print(len(img_fold))
    train_images = [x for x in images if x not in img_fold]
    #train_images = list(set(images)-set(img_fold))
    train_annotations = [x for x in annots if x not in annots_fold]
    #train_annotations = list(set(annots)- set(annots_fold))        
        
    # Split the dataset into train-valid-test splits 
    #train_images, val_images, train_annotations, val_annotations = train_test_split(img_fold, annots_fold, test_size = 0.2, random_state = 1)
    val_images, test_images, val_annotations, test_annotations = train_test_split(img_fold, annots_fold, test_size = 0.5)

    if not os.path.exists(f'{PATH}/cross_val_sph/dataset_fold_{n}'):
        os.makedirs(f'{PATH}/cross_val_sph/dataset_fold_{n}')
        os.makedirs(f'{PATH}/cross_val_sph/dataset_fold_{n}/train')
        os.makedirs(f'{PATH}/cross_val_sph/dataset_fold_{n}/test')
        os.makedirs(f'{PATH}/cross_val_sph/dataset_fold_{n}/val')

    print(train_images)
    # copy the splits into their folders
    copy_files_to_folder(train_images, f'{PATH}/cross_val_sph/dataset_fold_{n}/train')
    copy_files_to_folder(val_images, f'{PATH}/cross_val_sph/dataset_fold_{n}/val')
    copy_files_to_folder(test_images, f'{PATH}/cross_val_sph/dataset_fold_{n}/test')
    copy_files_to_folder(train_annotations, f'{PATH}/cross_val_sph/dataset_fold_{n}/train')
    copy_files_to_folder(val_annotations,  f'{PATH}/cross_val_sph/dataset_fold_{n}/val')
    copy_files_to_folder(test_annotations, f'{PATH}/cross_val_sph/dataset_fold_{n}/test')
