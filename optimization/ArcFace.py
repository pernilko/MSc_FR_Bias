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
import os
from sklearn import preprocessing

class ArcFace:

    def __init__(self):
        self.detector = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        self.detector.prepare(ctx_id=0, det_size=(640, 640))

        self.recognizer = insightface.model_zoo.get_model('buffalo_l')
        self.recognizer.prepare(ctx_id=0)
        

    '''
    Method loads jpg images from path
    Input: path -> specifies the path where the images are stored
    Return: returns an array containing all images loaded from the specified path
    '''
    def LoadData(self, path : str):
        testCounter = 0

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

    
    def ConvertToTorch(self, dataset, labels):
        # convert image list to numpy array
        new = []
        for img in dataset:
            img = np.transpose(img, (2, 0, 1))
            img = np.resize(img, (3, 98, 98))
            new.append(img)

        dataset = np.array(new)
        dataset = torch.from_numpy(dataset)
       
        
        # convert label list to tensor
        le = preprocessing.LabelEncoder()
        targets = le.fit_transform(labels)
        targets = torch.as_tensor(targets)

        data = torch.utils.data.TensorDataset(dataset.to(torch.float),targets)
        return data

    '''
    Method calculates similarity score between two images
    Input: inputImage1, inputImage2
    '''
    def CalculateSimilarityScores(self, a, inputImage1, inputImage2):
        #a = ArcFace.ArcFaceInit()

        # Retrieve faces
        face1 = a.detector.get(inputImage1)
        face2 = a.detector.get(inputImage2)

        # Feature Extraction
        featuresFace1 = a.recognizer.get(inputImage1, face1[0])
        featuresFace2 = a.recognizer.get(inputImage2, face2[0])

        # Regn ut similarity score
        similarityScore = a.recognizer.compute_sim(featuresFace1, featuresFace2)

        return similarityScore


    '''
    path = "/mnt/c/Users/PernilleKopperud/Documents/InfoSec/MasterThesis/master_thesis/MSc_FR_Bias/datasets/lfw"
    dataset = LoadData(path)
    for x in range(0,2,len(dataset)):
        for y in range(1,2,len(dataset)):
            similarityScore = CalculateSimilarityScores(x,y)
            print(similarityScore)
    '''