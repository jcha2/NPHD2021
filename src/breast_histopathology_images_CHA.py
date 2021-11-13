#!/usr/bin/env python
# coding: utf-8

# # Try using CoAtNet w/ Breast Histopathology Images

# * The original dataset consisted of 162 whole mount slide images of Breast Cancer (BCa) specimens scanned at 40x. From that, 277,524 patches of size 50 x 50 were extracted (198,738 IDC negative and 78,786 IDC positive). Each patch’s file name is of the format: uxXyYclassC.png — > example 10253idx5x1351y1101class0.png . Where u is the patient ID (10253idx5), X is the x-coordinate of where this patch was cropped from, Y is the y-coordinate of where this patch was cropped from, and C indicates the class where 0 is non-IDC and 1 is IDC.

# ## 1) Get Image Data & Data Preprocessing

# In[28]:


import pandas as pd
import numpy as np
import os
from tqdm.notebook import tqdm


# In[112]:


path = "/home/bis/211105_CSH_MPL/kaggleBreast/"

ids = os.listdir(path)
paths_1 = []
paths_0 = []
for id in tqdm(ids):
    try:
        files1 = os.listdir(path + id + '/1/')
        files0 = os.listdir(path + id + '/0/')
        for x in files1:
            paths_1.append(path + id + '/1/' + x)
        for x in files0:
            paths_0.append(path + id + '/0/' + x)
    except:
        FileNotFoundError
print(len(paths_1))
print(len(paths_0))
print(len(paths_0+paths_1))


# In[136]:


#train test split
#NPHD - positive 5000, negative 5000

import random

random.shuffle(paths_1)
random.shuffle(paths_0)
train_paths = paths_0[:200] + paths_1[:200]
valid_paths = paths_0[200:250]+paths_1[200:250]
test_paths = paths_0[250:500]+paths_1[250:500]
print(len(train_paths))
print(len(valid_paths))
print(len(test_paths))


# In[137]:


#image to array
#processing 277524 imgs take about 10 mins
#processing 10000 imgs take about 1 mins
import keras_preprocessing.image as IMAGE
import PIL
from PIL import Image
import torch
from torch.utils.data import  TensorDataset, DataLoader
import torchvision.transforms as transforms

transform = transforms.Compose([
    transforms.Resize([64, 64]),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

train_images=[]
for i in tqdm(train_paths):
    label = int(i[-5]) # class1.png --> 1
    img = transform(IMAGE.load_img(i, target_size=(64,64)))
    train_images.append((img,label))

valid_images=[]
for i in tqdm(valid_paths):
    label = int(i[-5]) # class1.png --> 1
    img = transform(IMAGE.load_img(i, target_size=(64,64)))
    valid_images.append((img,label))    

test_images=[]
for i in tqdm(test_paths):
    label = int(i[-5]) # class1.png --> 1
    img = transform(IMAGE.load_img(i, target_size=(64,64)))
    test_images.append((img,label))
    


# In[138]:


print("train img #",len(train_images))
print("test img #",len(test_images))


# In[139]:


from torch.utils.data import  TensorDataset, DataLoader

dataloaders = {}
dataloaders['train'] = DataLoader(train_images, batch_size=16, shuffle=True)
dataloaders['valid'] = DataLoader(valid_images, batch_size=len(valid_images), shuffle=True)
dataloaders['test'] = DataLoader(test_images, batch_size=len(test_images), shuffle=True)


# ## 2) Try CoAtNet

# def coatnet_0():<br/>
#     num_blocks = [2, 2, 3, 5, 2]            # L<br/>
#     channels = [64, 96, 192, 384, 768]      # D<br/>
#     return CoAtNet((224, 224), 3, num_blocks, channels, num_classes=1000)<br/>
# 
# <br/>
# def coatnet_1():<br/>
#     num_blocks = [2, 2, 6, 14, 2]           # L<br/>
#     channels = [64, 96, 192, 384, 768]      # D<br/>
#     return CoAtNet((224, 224), 3, num_blocks, channels, num_classes=1000)<br/>
# 
# <br/>
# def coatnet_2():<br/>
#     num_blocks = [2, 2, 6, 14, 2]           # L<br/>
#     channels = [128, 128, 256, 512, 1026]   # D<br/>
#     return CoAtNet((224, 224), 3, num_blocks, channels, num_classes=1000)<br/>
# 
# <br/>
# def coatnet_3():<br/>
#     num_blocks = [2, 2, 6, 14, 2]           # L<br/>
#     channels = [192, 192, 384, 768, 1536]   # D<br/>
#     return CoAtNet((224, 224), 3, num_blocks, channels, num_classes=1000)<br/>
# 
# 
# def coatnet_4():<br/>
#     num_blocks = [2, 2, 12, 28, 2]          # L<br/>
#     channels = [192, 192, 384, 768, 1536]   # D<br/>
#     return CoAtNet((224, 224), 3, num_blocks, channels, num_classes=1000)<br/>

# In[140]:


# Print total batches
total_batches = len(dataloaders["train"])
print(f"Total batches: {total_batches}")


# In[144]:


# Init CoAtNet model
import torch.nn as nn
from coatnet import coatnet_0,coatnet_1,coatnet_2,coatnet_3,coatnet_4
from coatnet import CoAtNet


#coatnet_4
num_blocks = [2, 2, 3, 5, 2]            # L
channels = [64, 96, 192, 384, 768]      # D
block_types=['C', 'C', 'T', 'T']        # 'C' for MBConv, 'T' for Transformer
model = CoAtNet((64,64), 3, num_blocks, channels, num_classes = 2) 

#loss - cross entropy
#optimizer - adam
# loss_fn = nn.MSELoss()
loss_fn = nn.CrossEntropyLoss()

# Define Optimizer - 이해 필요
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.0001,
    betas=(0.9, 0.999),
    eps=10e-8,
    amsgrad=False,
)

# Print model info
print(model)


# In[145]:


def div(i1, i2):
    return i1/i2 if i2 else 0


# In[146]:


# Train CoAtNet
train = {'loss':[]}
valid = {'loss':[],'accuracy':[],'specificity':[],'recall':[],'precision':[],'negpred':[],'f1':[]}

epochs = 20 
for eid in range(epochs):
    # Logging
    print("Epoch {}".format(eid))
    loss_avg = 0
    for i, (inputs, targets) in enumerate(dataloaders["train"]):
        # Train model
        model.train()
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        loss_avg += loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()   
        # Logging
    train['loss'].append(loss_avg/total_batches)
    print("Training loss: {}".format(loss_avg/total_batches))
    # Validate model
    model.eval()
    with torch.no_grad():
        TP = 0
        FP = 0
        TN = 0
        FN = 0
        total = 0
        loss_avg = 0
        for i, (inputs, targets) in enumerate(dataloaders["valid"]):
            outputs = model(inputs)
            loss_avg += loss_fn(outputs, targets)
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            TP += ((predicted == targets)&(predicted == 1)).sum().item()
            FP += ((predicted != targets)&(targets == 0)).sum().item()
            TN += ((predicted == targets)&(predicted == 0)).sum().item()
            FN += ((predicted != targets)&(targets == 1)).sum().item()
        acc = div((TP+TN),total)
        spec = div(TN,(TN+FP))
        negpred = div(TN,(TN+FN))
        rec = div(TP,(TP+FN))
        prec = div(TP,(TP+FP))
        f1 = div(2*(prec*rec),(prec+rec))
        valid['accuracy'].append(acc) 
        valid['specificity'].append(spec) 
        valid['recall'].append(rec) 
        valid['precision'].append(prec) 
        valid['negpred'].append(negpred) 
        valid['f1'].append(f1) 
        valid['loss'].append(loss_avg/total_batches) 
        print("Validation loss: {}".format(loss_avg/total_batches))
        print("Validation accuracy: {}".format(acc))
        


# In[147]:


TP = 0
FP = 0
TN = 0
FN = 0
total = 0
for i, (inputs, targets) in enumerate(dataloaders["test"]):
    outputs = model(inputs)
    _, predicted = torch.max(outputs.data, 1)
    total += targets.size(0)
    correct += (predicted == targets).sum().item()
    TP += ((predicted == targets)&(predicted == 1)).sum().item()
    FP += ((predicted != targets)&(targets == 0)).sum().item()
    TN += ((predicted == targets)&(predicted == 0)).sum().item()
    FN += ((predicted != targets)&(targets == 1)).sum().item()
acc = div((TP+TN),total)
spec = div(TN,(TN+FP))
negpred = div(TN,(TN+FN))
rec = div(TP,(TP+FN))
prec = div(TP,(TP+FP))
f1 = div(2*(prec*rec),(prec+rec))


# In[148]:

print("Test accuracy: {}".format(round(acc,10)))
print("Test specifity: {}".format(round(spec,10)))
print("Test negative predicable: {}".format(round(negpred,10)))
print("Test recall: {}".format(round(rec,10)))
print("Test precision: {}".format(round(prec,10)))
print("Test f1: {}".format(round(f1,10)))
print("TP:",TP)
print("FP:",FP)
print("TN:",TN)
print("FN:",FN)

