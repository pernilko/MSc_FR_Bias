import os
import pandas as pd
from torchvision.io import read_image
import torch
from torch.utils.data import DataLoader

def load_dataset(path : str):

    labels = []
    dataset = []

    for subdir, dirs, files in os.walk(path):
        if (testCounter > 100):
            return dataset, labels
        for file in files:
            completePath = os.path.join(subdir, file)
            image = ins_get_image(os.path.splitext(completePath)[0], os.path.splitext(file)[0])
            dataset.append(image)
            name = os.path.splitext(file)[0]
            labels.append(name)
            testCounter = testCounter + 1
    
    return dataset, labels

def data_preprocessing():
    
    return


# Filter out unwanted images based on quality, or other factors such as wearing glasses etc