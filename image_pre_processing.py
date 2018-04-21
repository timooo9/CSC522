import pandas as pd
import numpy as np
import os
import cv2
import torch
from skimage.feature import CENSURE

df = pd.DataFrame(columns=['image','breed'])
directoryTrainPath = 'Path/data/trainBreed'
directoryTrain = os.listdir(directoryTrainPath)
breedLabels = pd.read_csv('/Path/data/preTrainLabels/trainLabelsBreed.csv')
breedInts = pd.read_csv('/Path/data/preTrainLabels/breedsToInts.csv')

count = 0
#read each image
for i in range(len(directoryTrain)):
    filename = directoryTrain[i]
    filePath = '/Users/enyalos/522/project/project/data/trainBreed/' + filename
    #read in image and resize and turn to grey scale 0-255
    img = cv2.imread(filePath,0)
    img = cv2.resize(img, (150,150))

    #Identify keypoints
    detector = CENSURE()
    detector.detect(img)
    keypoint = detector.keypoints

    #mask the image - all pixels less than 40 change to 255
    mask = img < 40
    img[mask] = 255

    #add keypoints as black squares back into image
    for point in range(len(keypoint)):
        row = keypoint[point][0]
        column = keypoint[point][1]
        for i in range(3):
            print(i)
            img[row + 1][column+i] = 0
            img[row + 2][column+i] = 0
            img[row + 3][column+i] = 0
            img[row + 4][column+i] = 0

    #make mirror copy of image
    mirrorImage = img[:, ::-1]

    img = img.tolist()
    mirrorImage = mirrorImage.tolist()

    splitFileName = filename.split('.')
    index = list(breedLabels.index[breedLabels['id'] == splitFileName[0]])[0]
    breedClass = breedLabels.loc[index,'breed']
    breedDF = list(breedInts.index[breedInts['breed'] == breedClass])[0]
    breed = breedInts.loc[breedDF, 'id']
    df.loc[count, 'breed'] = breed
    df.loc[count, 'image'] = img
    count = count + 1
    df.loc[count, 'breed'] = breed
    df.loc[count, 'image'] = mirrorImage
    count = count + 1

dogLabels = pd.DataFrame(columns=['breed'])
dogLabels['breed'] = df['breed']
dogLabels.to_csv("trainBreedLabels.csv", index=False)

#save images into pytorch tensors.
trainTensor = torch.Tensor(df['image'])
trainTensor = torch.unsqueeze(trainTensor, 1)
torch.save(trainTensor, 'trainTensor.pt')
