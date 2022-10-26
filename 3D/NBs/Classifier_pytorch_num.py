#!/usr/bin/env python
# coding: utf-8

import os, shutil, pathlib
import torch
#torch 1.10.0
import pandas as pd

from torch import nn
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from torch.utils.data import random_split
from torchvision import models
from datetime import datetime

import numpy as np
import time

import matplotlib.pyplot as plt
#matplotlib 3.3.3
import sklearn
from sklearn.datasets import make_classification
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
#sklearn 0.23.2

casez = []
casez = np.append(casez,"W15.0")
# casez = np.append(casez,"W15.25")
# casez = np.append(casez,"W15.5")
# casez = np.append(casez,"W15.75")
# casez = np.append(casez,"W16.0")
# casez = np.append(casez,"W16.2")
# casez = np.append(casez,"W16.3")
# casez = np.append(casez,"W16.4")
casez = np.append(casez,"W16.5")
# casez = np.append(casez,"W16.6")
# casez = np.append(casez,"W16.7")
# casez = np.append(casez,"W16.8")
# casez = np.append(casez,"W17.0")
# casez = np.append(casez,"W17.25")
# casez = np.append(casez,"W17.5")
# casez = np.append(casez,"W17.75")
casez = np.append(casez,"W18.0")


c = []
c = np.append(c,15)
# c = np.append(c,15.25)
# c = np.append(c,15.5)
# c = np.append(c,15.75)
# c = np.append(c,16)
# c = np.append(c,16.2)
# c = np.append(c,16.3)
# c = np.append(c,16.4)
c = np.append(c,16.5)
# c = np.append(c,16.6)
# c = np.append(c,16.7)
# c = np.append(c,16.8)
# c = np.append(c,17)
# c = np.append(c,17.25)
# c = np.append(c,17.5)
# c = np.append(c,17.75)
c = np.append(c,18)


path = pathlib.Path("/home/physics/phubdf/Numerical_Data")
os.chdir(path)

if os.path.exists(f"{path}/labels"):
    shutil.rmtree(f"{path}/labels")
os.mkdir(f"{path}/labels")
for i in range(0,len(casez)):
    csv_input = pd.read_csv(f'{path}/{casez[i]}/labels.csv')
    csv_input.replace(to_replace=0,value=i,inplace = True)
    csv_input.to_csv(f'{path}/labels/labels{c[i]}.csv', index=False)


src = os.listdir(f'{path}/labels')
a = pd.concat([pd.read_csv(f'{path}/labels/{file}') for file in src ], ignore_index=True)
a.to_csv(f'{path}/labels/labels.csv', index=False)




class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = np.loadtxt(f"{img_path}")
        image = np.square(image)
        image = image.reshape(1,30,30,30)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label




batch_size = 32
ndata = 4000
for i in range(0,len(casez)):
    if i == 0:
        data = CustomImageDataset(annotations_file=f"{path}/labels/labels{c[i]}.csv",img_dir=f"{path}/{casez[i]}")
        training_data, validation_data, test_data = random_split(data,[int(ndata*0.8),int(ndata*0.15),int(ndata*0.05)])
    else:
        data = CustomImageDataset(annotations_file=f"{path}/labels/labels{c[i]}.csv",img_dir=f"{path}/{casez[i]}")
        train_set, validation_set, test_set = random_split(data,[int(ndata*0.8),int(ndata*0.15),int(ndata*0.05)]) 
        training_data = ConcatDataset([training_data,train_set])
        validation_data = ConcatDataset([validation_data,validation_set])
        test_data = ConcatDataset([test_data,test_set])
        

print(len(training_data))
print(len(validation_data))
print(len(test_data))



# Create data loaders.

train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
validation_dataloader = DataLoader(validation_data, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)




# train_features, train_labels = next(iter(train_dataloader))
# print(f"Feature batch shape: {train_features.size()}")
# print(train_features)
# print(f"Labels batch shape: {train_labels.size()}")
# img = train_features[0].squeeze()
# label = train_labels
# print(f"Label: {label}")



# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

model = models.video.r3d_18()
model.stem[0] = nn.Conv3d(in_channels=1, out_channels=64, kernel_size=(3,7,7), stride=(1,2,2), padding=0, bias=False)
model.fc = nn.Linear(in_features=512,out_features=len(c),bias=True)
if 0 != 0:
	for i in range(0,):
		if os.path.exists(f"{path}/Numerical_Data/saved models/saved_model[{i+1}].pth"):
					model.load_state_dict(torch.load(f"{path}/Numerical_Data/saved models/saved_model[{i+1}].pth"))
if torch.cuda.is_available():
    model.cuda()
print(model)



def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")



loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)


epochs = 10
if os.path.exists(f"{path}/Numerical_Data/saved models"):
    shutil.rmtree(f"{path}/Numerical_Data/saved models")
os.mkdir(f"{path}/Numerical_Data/saved models")
min_valid_loss = np.inf
start = time.time()
tl = np.array([])
vl = np.array([])

for e in range(epochs):
    st = time.time()
    train_loss = 0.0
    model.train()     # Optional when not using Model Specific layer
    for data, labels in train_dataloader:
        if torch.cuda.is_available():
            data, labels = data.cuda(), labels.cuda()
        
        optimizer.zero_grad()
        target = model(data.float())
        loss = loss_fn(target,labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    
    valid_loss = 0.0
    model.eval()     # Optional when not using Model Specific layer
    for data, labels in validation_dataloader:
        if torch.cuda.is_available():
            data, labels = data.cuda(), labels.cuda()
        
        target = model(data.float())
        loss = loss_fn(target,labels)
        valid_loss = loss.item() * data.size(0)
    
    et = time.time()
    rt = et-st

    print(f'Epoch {e+1} \t Runtime: {round(rt,2)}s \t Training Loss: {train_loss / len(train_dataloader)} \t Validation Loss: {valid_loss / len(validation_dataloader)}')
    if min_valid_loss > valid_loss:
        print(f'Validation Loss Decreased({min_valid_loss:.6f}--->{valid_loss:.6f}) \t Saving The Model')
        min_valid_loss = valid_loss
        # Saving State Dict
        torch.save(model.state_dict(), f"{path}/Numerical_Data/saved models/saved_model[{e+1}].pth")
    else:
        print(" ")
    
    tl = np.append(tl, train_loss / len(train_dataloader))
    vl = np.append(vl,valid_loss / len(validation_dataloader))

end = time.time()
total = end-start
print(f"Total runtime: {round(total,2)}s")


x = np.arange(0,epochs,1)
plt.plot(x+1, tl, "b+", label="Training loss")
np.savetxt(f"tl-C{len(c)}-D30-{datetime.now()}.csv", tl, delimiter=",")
plt.plot(x+1, vl, "ro", label="Validation loss")
np.savetxt(f"vl-C{len(c)}-D30-{datetime.now()}.csv", vl, delimiter=",")
plt.title("Training and validation loss")
plt.xlabel("epochs")
plt.legend()
plt.show()




predict = []
p = []
model.eval()
for i in range(0,200*len(c)):
    x, y = test_data[i][0], test_data[i][1]
    x = x.reshape(1,1,30,30,30)
    x = torch.from_numpy(x)
    x = x.float()
    with torch.no_grad():
        pred = model (x.cuda()) if torch.cuda.is_available() else model(x)
        predicted, actual = pred[0].argmax(0), y 
        predicted = torch.Tensor.cpu(predicted)
        predict = np.append(predict, predicted)
        p = np.append(p, actual)
      


cm = confusion_matrix(p, predict)
print(cm)
np.savetxt(f"cm-C{len(c)}-D30-{datetime.now()}.csv", cm, delimiter=",")
score = round(accuracy_score(p, predict)*100,2)
print(f"Model Accuracy: {score}%")




