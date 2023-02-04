import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image
import torch
import torchvision
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from optimization.ArcFace import ArcFace 
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch import linalg
from insightface2.recognition.arcface_torch.backbones import iresnet50
import os
from torchvision import datasets, transforms
from data.synthetic_data_generation import load_pretrained_model


a = ArcFace()
path = "/mnt/c/Users/PernilleKopperud/Documents/InfoSec/MasterThesis/master_thesis/MSc_FR_Bias/src/datasets/lfw"
#dataset = a.LoadData(path)
#for x in range(0,len(dataset)-1):
#    similarityScore = a.CalculateSimilarityScores(a,dataset[x],dataset[x+1])
#    print(similarityScore)

loadedModel = torch.load("backbone.pth",  map_location=torch.device('cpu'))
model : torch.nn.Module = iresnet50()
model.load_state_dict(loadedModel, strict=True)
model.eval()

params = model.state_dict()
keys = list(params.keys())

# freeze layers
for param in model.parameters():
    param.requires_grad = False

# unfreeze
frozenParams = ['conv1.weight', 'bn1.weight', 'bn1.bias',  'prelu.weight']
frozenLayers = ['layer1', 'layer2']

for name, param in model.named_parameters():
    does_not_require_grad = not param.requires_grad
    param_not_in_name = all((not x == name) for x in frozenParams )
    layer_not_in_name =  all((not x in name) for x in frozenLayers)

    if (does_not_require_grad and ( param_not_in_name and layer_not_in_name)):
        param.requires_grad = True
        #print(name)



    #print(param.requires_grad)
transform = transforms.Compose([transforms.Resize(255),
transforms.CenterCrop(224),
transforms.ToTensor()])
d = datasets.ImageFolder(path, transform=transform)

dataset, labels = a.LoadData(path) #ArcFace.LoadData(a,path) # Tuple
data = a.ConvertToTorch(dataset, labels)
length = [int(len(data)*0.8), int(len(data)*0.2)]

# data 

train_dataset, val_dataset = torch.utils.data.random_split(data, length)

'''
normalized_train_dataset = []
for tensor in train_dataset:
    normalizedTensor = torch.nn.functional.normalize(torch.Tensor(tensor), p=2.0, dim = 1)
    normalized_train_dataset.append(normalizedTensor)

normalized_val_dataset = []
for tensor in val_dataset:
    normalizedTensor = torch.nn.functional.normalize(torch.Tensor(tensor), p=2.0, dim = 1)
    normalized_val_dataset.append(normalizedTensor)
'''



#
training_loader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=2)

validation_loader = torch.utils.data.DataLoader(val_dataset, batch_size=4, shuffle=True, num_workers=2)

optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
loss_fn = torch.nn.CrossEntropyLoss()


def train_one_epoch(epoch_index):
    running_loss = 0.
    last_loss = 0.

    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    for i, data in enumerate(training_loader):
        # Every data instance is an input + label pair
        inputs, labels = data

        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        outputs = model(inputs)

        # Compute the loss and its gradients
        loss = loss_fn(outputs, labels)
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()
        if i % 1000 == 999:
            last_loss = running_loss / 1000 # loss per batch
            print('  batch {} loss: {}'.format(i + 1, last_loss))
            running_loss = 0.

    return last_loss

epoch_number = 0

EPOCHS = 5

best_vloss = 1_000_000.

for epoch in range(EPOCHS):
    print('EPOCH {}:'.format(epoch_number + 1))

    # Make sure gradient tracking is on, and do a pass over the data
    model.train(True)
    avg_loss = train_one_epoch(epoch_number)

    # We don't need gradients on to do reporting
    model.train(False)

    running_vloss = 0.0
    for i, vdata in enumerate(validation_loader):
        vinputs, vlabels = vdata
        voutputs = model(vinputs)
        vloss = loss_fn(voutputs, vlabels)
        running_vloss += vloss

    avg_vloss = running_vloss / (i + 1)
    print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))

    # Track best performance, and save the model's state
    if avg_vloss < best_vloss:
        best_vloss = avg_vloss
        model_path = 'model_{}'.format(epoch_number)
        torch.save(model.state_dict(), model_path)

    epoch_number += 1


# https://jimmy-shen.medium.com/pytorch-freeze-part-of-the-layers-4554105e03a6