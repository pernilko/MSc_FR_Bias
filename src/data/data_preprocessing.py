import os
import pandas as pd
from torchvision.io import read_image
import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from matplotlib import pyplot as plt
import numpy as np

def load_dataset(img_path : str):
    
    tsfm = transforms.Compose([
        transforms.ToTensor()
])
    data = datasets.ImageFolder(img_path, tsfm)
    dataloader = DataLoader(data, batch_size=20)
    for batch in dataloader:
        inputs, targets = batch
        for img in inputs:
            image  = img.cpu().numpy()
            # transpose image to fit plt input
            image = image.T
            # normalise image
            data_min = np.min(image, axis=(1,2), keepdims=True)
            data_max = np.max(image, axis=(1,2), keepdims=True)
            scaled_data = (image - data_min) / (data_max - data_min)
            # show image
            plt.imshow(scaled_data)
            plt.show()

    return dataloader

def data_preprocessing():
    
    return


load_dataset("/mnt/c/Users/PernilleKopperud/Documents/InfoSec/MasterThesis/master_thesis/MSc_FR_Bias/src/datasets/lfw")
