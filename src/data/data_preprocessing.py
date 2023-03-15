import os
import pandas as pd
from torchvision.io import read_image
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets
from matplotlib import pyplot as plt
import numpy as np
import cv2
import shutil

def load_dataset(img_path : str, batch_size : int, transforms):
    
    data = datasets.ImageFolder(img_path, transforms)
    '''
    for c in data.targets:
        print(c)
    '''
   

    length = [round(len(data)*0.8), round(len(data)*0.2)]
    train_data, val_data = torch.utils.data.random_split(data, length)

    training_data_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=2)
    validation_data_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=2)
    
    '''
    for batch in training_data_loader:
        inputs, targets = batch
        print(targets)
    '''
    
    

    return training_data_loader, validation_data_loader

'''
Method loads test dataset

Parameters:
    img_path (str) : path to folder containing dataset
    batch_size (int) : batch size
    transforms () : 
    fgnet_organize (bool) : boolean indicating if the FG-NET dataset should be organized into one folder per identity 
Return:
    test_data_loader (DataLoader)

'''
def load_test_dataset(img_path : str, batch_size : int, transforms):
    data = datasets.ImageFolder(img_path, transforms)
    #classes = data.imgs.
    #print(classes)
    test_data_loader = DataLoader(data, batch_size=batch_size, shuffle=False, num_workers=2)
    return test_data_loader


def test_image_loader(training_data_loader, validation_data_loader):
    for train_batch, val_batch in zip(training_data_loader, validation_data_loader):
        train_inputs, train_targets = train_batch
        val_inputs, val_targets = val_batch
        counter = 0
        for train_img, val_img in zip(train_inputs, val_inputs):
            train_image  = train_img.cpu().numpy()
            val_image  = val_img.cpu().numpy()
            # transpose image to fit plt input
            train_image = train_image.T
            val_image = val_image.T

            # normalise image
            train_data_min = np.min(train_image, axis=(1,2), keepdims=True)
            train_data_max = np.max(train_image, axis=(1,2), keepdims=True)
            train_scaled_data = (train_image - train_data_min) / (train_data_max - train_data_min)

             # normalise image
            val_data_min = np.min(train_image, axis=(1,2), keepdims=True)
            val_data_max = np.max(train_image, axis=(1,2), keepdims=True)
            val_scaled_data = (val_image - val_data_min) / (val_data_max - val_data_min)

            # show image
            plt.imshow(train_scaled_data)
            plt.savefig("train_" + str(counter)+ ".png")
            plt.imshow(val_scaled_data)
            plt.savefig("val" + str(counter)+ ".png")
            counter = counter + 1

'''
Method organizes FG-NET into one folder per identity

Parameters:
    data_dir (str) : path to folder containing FG-NET dataset
Return:
    None. 
'''
def orgranize_fgnet_dataset(data_dir):
     
    identities = []
    for filename in os.listdir(data_dir):
            if not filename.endswith('.JPG'):
                continue
            
            fname = filename.split('.')[0] # remove .jpg
            
            name = fname.split('A')[0] # name of identity
            if name not in identities:
                identities.append(name)
                os.makedirs(f"{data_dir}/{name}/", exist_ok=True)
            
            age = fname.split('A')[1] # age
            shutil.move(f"{data_dir}/{filename}", f"{data_dir}/{name}/{name}_{age}.jpg")


