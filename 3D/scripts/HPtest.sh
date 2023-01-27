#!/bin/bash
getseed=${1:-"N"}
no=${2:-4000}
size=${3:-30}
epochs=${4:-10}
re=${5:-0}

echo "Version:" `git describe --tags --long` echo "Branch:" `git branch --show-current`
echo $getseed $no $size $epochs $re


workdir=$(pwd)
mkdir -p $workdir/N$no-L$size/HP
cd ../
numdir=$(pwd)/Numerical_Data
fdir=$(pwd)/NBs
workdir=$workdir/N$no-L$size/HP
echo $numdir
echo $workdir

cd $workdir

job=`printf "$fdir/HP-N$no-L$size.sh"`
py=`printf "$fdir/HP-N$no-L$size.py"`
echo $py

now=$(date +"%T")
echo "Current time : $now"






cat > ${py} << EOD
#!/usr/bin/env python
# coding: utf-8


import os, shutil, pathlib
import torch
#torch 1.7.1
import pandas as pd
from functools import partial
from torch import nn
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from torch.utils.data import random_split
from torchvision import models
from datetime import datetime


import numpy as np
import time
import random

import matplotlib.pyplot as plt
#matplotlib 3.3.3
import sklearn
from sklearn.datasets import make_classification
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
#sklearn 0.23.2


import torch.nn.functional as F
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler

casez = []
casez = np.append(casez,"W15.0")
casez = np.append(casez,"W15.25")
casez = np.append(casez,"W15.5")
casez = np.append(casez,"W15.75")
casez = np.append(casez,"W16.0")
casez = np.append(casez,"W16.2")
casez = np.append(casez,"W16.3")
casez = np.append(casez,"W16.4")
casez = np.append(casez,"W16.5")
casez = np.append(casez,"W16.6")
casez = np.append(casez,"W16.7")
casez = np.append(casez,"W16.8")
casez = np.append(casez,"W17.0")
casez = np.append(casez,"W17.25")
casez = np.append(casez,"W17.5")
casez = np.append(casez,"W17.75")
casez = np.append(casez,"W18.0")


c = []
c = np.append(c,15)
c = np.append(c,15.25)
c = np.append(c,15.5)
c = np.append(c,15.75)
c = np.append(c,16)
c = np.append(c,16.2)
c = np.append(c,16.3)
c = np.append(c,16.4)
c = np.append(c,16.5)
c = np.append(c,16.6)
c = np.append(c,16.7)
c = np.append(c,16.8)
c = np.append(c,17)
c = np.append(c,17.25)
c = np.append(c,17.5)
c = np.append(c,17.75)
c = np.append(c,18)

path = pathlib.Path("$numdir")
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
print("created labels")



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





# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

model = models.video.r3d_18()
model.stem[0] = nn.Conv3d(in_channels=1, out_channels=64, kernel_size=(3,7,7), stride=(1,2,2), padding=0, bias=False)
model.fc = nn.Linear(in_features=512,out_features=len(c),bias=True)

if torch.cuda.is_available():
    model.cuda()
print(model)


loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)


def train(model, optimizer, train_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.train()
    train_loss = 0.0
    for data, labels in train_loader:
        if torch.cuda.is_available():
            data, labels = data.cuda(), labels.cuda()
        optimizer.zero_grad()
        target = model(data.float())
        loss = loss_fn(target,labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()


def test(model, data_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(data_loader):
            data, target = data.to(device), target.to(device)
            data = data.float()
            target = target.float()
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    return correct / total

ndata = $no
def train_t(config):
    # Data Setup
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

    batch_size=config["batch_size"]
    train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
    validation_dataloader = DataLoader(validation_data, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

    model = models.video.r3d_18()
    model.stem[0] = nn.Conv3d(in_channels=1, out_channels=64, kernel_size=(3,7,7), stride=(1,2,2), padding=0, bias=False)
    model.fc = nn.Linear(in_features=512,out_features=len(c),bias=True)

    if torch.cuda.is_available():
        model.cuda()

    optimizer = torch.optim.SGD(
        model.parameters(), lr=config["lr"], momentum=config["momentum"])
    for i in range(5):
        train(model, optimizer, train_dataloader)
        acc = test(model, test_dataloader)

        # Send the current training result back to Tune
        tune.report(mean_accuracy=acc)

        if i % 5 == 0:
            # This saves the model to the trial directory
            torch.save(model.state_dict(), "./model.pth")




config = {
    "lr": tune.loguniform(1e-4, 1e-1),
    "batch_size": tune.choice([2, 4, 8, 16, 32, 64]),
    "momentum": tune.uniform(0.1, 0.9)
}


tuner = tune.run(
    	train_t,
    	config=config,
	num_samples=10,
	)
results = tuner.fit()
logdir = results.get_best_result("mean_accuracy", mode="max").log_dir
print("Best trial config: {}".format(logdir.config))
print("Best trial final accuracy: {}".format(
	logdir.last_result["mean_accuracy"]))


dfs = {result.log_dir: result.metrics_dataframe for result in results}


EOD


cat > ${job} << EOD
#!/bin/bash
#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=3700
#SBATCH --time=48:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:quadro_rtx_6000:2

module purge
module restore PT
module list


srun $py


EOD

chmod 755 ${job}
chmod g+w ${job}
chmod 755 ${py}

sbatch ${job}

