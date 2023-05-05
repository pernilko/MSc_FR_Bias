import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from matplotlib import pyplot as plt
import numpy as np
import shutil

'''
Method loads dataset and splits it into a training dataset and a validation dataset

Parameters:
    img_path (str) : path to training/validation dataset
    batch_size (int) : the size of the batches
    transforms (transforms) : transformations that are to be made to the dataset
Return:
    training_data_loader (DataLoader) : dataloader for the training dataset
    validation_data_loader (DataLoader) : dataloader for the validation dataset
'''
def load_dataset(img_path : str, batch_size : int, transforms : transforms):
    
    data = datasets.ImageFolder(img_path, transforms)
   
    length = [round(len(data)*0.8), round(len(data)*0.2)]
    train_data, val_data = torch.utils.data.random_split(data, length)

    training_data_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=0)
    validation_data_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=0)
    
    return training_data_loader, validation_data_loader

'''
Method loads test dataset

Parameters:
    img_path (str) : path to folder containing test dataset
    batch_size (int) : the size of the batches
    transforms (transforms) : transformations that are to be made to the dataset
Return:
    test_data_loader (DataLoader) : dataloader for the test dataset

'''
def load_test_dataset(img_path : str, batch_size : int, transforms : transforms):
    data = datasets.ImageFolder(img_path, transforms)
    test_data_loader = DataLoader(data, batch_size=batch_size, shuffle=False, num_workers=2)

    return test_data_loader

'''
Method plots images in training and validation loaders (for testing purposes)

Parameters:
    training_data_loader (DataLoader) : dataloader containing the training dataset
    validation_data_loader (DataLoader) :  dataloader containing the validation dataset
Return:
    None.
'''
def test_image_loader(training_data_loader : DataLoader, validation_data_loader : DataLoader):
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
Method organizes FG-NET into one folder per identity. 
Method is intended to be used when the dataset is initially downloaded as all images are in one folder 
where the name of the file indicates identity. To be able to input the data into the model, it has to be organized into a folder per identity.

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

def organize_dataset(data_dir : str):
    identities = []
    for filename in os.listdir(data_dir):
        if not filename.endswith('.jpg'):
            continue
        
        fname = filename.split('.')[0] # remove .jpg

        name = fname.split('_')[0] # name of identity
        if name not in identities:
            identities.append(name)
            os.makedirs(f"{data_dir}/{name}/", exist_ok=True)
        
        fname_age_male = fname.split('_')[1].split('m')
        fname_age_female = fname.split('_')[1].split('f')

        age = 0
        if len(fname_age_female) == 2:
            age = fname_age_female[1]
        elif len(fname_age_male) == 2:
            age = fname_age_male[1]
        else:
            raise Exception("Could not find age")

        shutil.move(f"{data_dir}/{filename}", f"{data_dir}/{name}/{name}_{age}.jpg")

def count_number_of_images(data_dir : str):
    num_of_imgs = sum([len(files) for r, d, files in os.walk(data_dir)])
    return num_of_imgs

def sort_filenames(filenames):
    print(filenames)
    list = []
    for i in range(len(filenames)):
        if(len(list) == 0):
            list.append(filenames[i])
        else:
            age = getage(filenames[i])
            previous_age = getage(list[i-1])
            if age > previous_age:
                list.append(age)
            else:
                list.insert(list.index(list[i-1]), filenames[i])
    return list


def getage(filename):
    identity_age = filename.split('.')[0]
    age = identity_age.split('_')[1]
    return age


def plot_mixed_dataset_images(imgs_path):

    for directory in os.listdir(imgs_path):
        dir = os.path.join(imgs_path, directory)

        filenames = []
        for filename in sorted(os.listdir(dir)):
            filenames.append(filename)

        sorted_filenames = sort_filenames(filenames)
        
        ages = []
        imgs = []
        num_of_imgs = 0
        for filename in sorted_filenames:
            print(filename)
            f = os.path.join(dir, filename)
            identity_age = filename.split('.')[0]
            age = identity_age.split('_')[1]
            ages.append(age)
            img = plt.imread(f)
            imgs.append(img)
            num_of_imgs += 1
            
        
        fig, axs = plt.subplots(1, num_of_imgs, figsize=(num_of_imgs*4,4), dpi=100)
        for img, ax, age in zip(imgs, axs, ages):
            ax.imshow(img)
            ax.set_title(f'Label "{str(age)}"')
        
        os.makedirs("mixed_dataset/plots", exist_ok=True)
        # Display the plot
        plt.savefig(f"mixed_dataset/plots/{directory}.pdf")


plot_mixed_dataset_images("datasets/mixed_dataset_v2/")

