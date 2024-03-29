#!/bin/bash
getseed=${1:-"N"} #Set to Y if you what to reuse stored seed
epochs=${2:-50} #No of epochs
re=${3:-0} #Set restart point
no=${4:-4000} #Number of smaples to take from each category
size=${5:-30} #System size used
imgsize=${6:-100} #image size used
categories=${7:-"15,15.25,15.5,15.75,16,16.2,16.3,16.4,16.5,16.6,16.7,16.8,17,17.25,17.5,17.75,18"} #List of categories to use separated by ,

echo $getseed $epochs $re $no $size $imgsize $categories 


#execute file in terminal while in the output folder
workdir=$(pwd)

cd ../
strdir=$(pwd)
cd ../../
numdir=$(pwd)/I5000D"$size"s"$imgsize"
echo $numdir
fdir=$strdir/NBs
sdir=$strdir/scripts
IFS=', ' read -r -a array <<< $categories
classes=${#array[@]}
mkdir -p $workdir/I$no-L$size-$classes-s$imgsize
workdir=$workdir/I$no-L$size-$classes-s$imgsize
echo $numdir
echo $workdir

cd $workdir

job=`printf "$fdir/Img-N$no-L$size-$classes-s$imgsize.sh"`
py=`printf "$fdir/Img-N$no-L$size-$classes-s$imgsize.py"`
echo $py

now=$(date +"%T")
echo "Current time : $now"





cat > ${py} << EOD
#!/usr/bin/env python
# coding: utf-8
print("--> importing modules")
import os, shutil, pathlib
import torch
print("--> torch version used = 1.7.1, version loaded = " + str(torch.__version__))
import pandas as pd
from torch import nn
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from torch.utils.data import random_split
from torchvision import models

from datetime import datetime
import numpy as np
import pickle
import time
import random

import matplotlib
import matplotlib.pyplot as plt
print("--> matplotlib version used = 3.3.3, version loaded = " + str(matplotlib.__version__))

import sklearn
from sklearn.datasets import make_classification
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

import PIL
from PIL import Image
print("--> Pillow version used = 8.0.1, version loaded = " + str(PIL.__version__))

print("--> sklearn version used = 0.23.2, version loaded = " + str(sklearn.__version__))
print("--> import complete")
print(datetime.now())
print("$getseed $epochs $re $no $size")

class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file, nrows=$no)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
    def __len__(self):
        return len(self.img_labels)
    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = Image.open(img_path)
        image = np.asarray(image)
        image = np.moveaxis(image, -1, 0)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
print("--> defining categories")
c = np.fromstring("$categories",dtype=float,sep=",")
print(c)
casez = []
for i in range (0, len(c)):
    casez = np.append(casez, "W"+str(c[i]))
print(casez)
print("--> categories have been defined. No. of categories used = " + str(len(c)))
store="N$no-L$size"
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
if "$getseed" != "N":
        print("--> loading previous trial seed")
        f = open("$workdir/lastseed.txt", "r")
        seed = int(f.read())
        torch.manual_seed(seed)
        random.seed(seed)
        f.close()
        print("--> seed loaded")
else:
        seed = torch.seed()
        random.seed(seed)
f = open("$workdir/lastseed.txt", "w")
f.write(str(seed))
print("current seed: " + str(seed))
f.close()
print("--> seed saved to lastseed.txt in $workdir")
path = pathlib.Path("$numdir")
os.chdir(path)
print("--> creating labels file to identify files to be used")
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
print("--> created labels file")
###################################
print("--> creating datasets for usage for training, validation and testing")
batch_size = 32
ndata = $no
for i in range(0,len(casez)):
    if i == 0:
        data = CustomImageDataset(annotations_file=f"{path}/labels/labels{c[i]}.csv",img_dir=f"{path}/{casez[i]}")
        print(len(data))
        training_data, validation_data, test_data = random_split(data,[int(ndata*0.8),int(ndata*0.15),int(ndata*0.05)])
    else:
        data = CustomImageDataset(annotations_file=f"{path}/labels/labels{c[i]}.csv",img_dir=f"{path}/{casez[i]}")
        train_set, validation_set, test_set = random_split(data,[int(ndata*0.8),int(ndata*0.15),int(ndata*0.05)]) 
        training_data = ConcatDataset([training_data,train_set])
        validation_data = ConcatDataset([validation_data,validation_set])
        test_data = ConcatDataset([test_data,test_set])
        
print("--> created datasets")
print("--> training set contains " + str(len(training_data)) + " files")
print("--> validation set contains " + str(len(validation_data)) + " files")
print("--> test set contains " + str(len(test_data)) + " files")
# Create data loaders.
train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
validation_dataloader = DataLoader(validation_data, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
train_features, train_labels = next(iter(train_dataloader))
print(f"Feature batch shape: {train_features.size()}")
print(f"Labels batch shape: {train_labels.size()}")
img = train_features[0].squeeze()
label = train_labels
print(f"Label: {label}")
# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")
print("--> preparing model from resnet18 network")
model = models.resnet18()
model.conv1 = nn.Conv2d(4, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
model.fc = nn.Linear(in_features=512,out_features=len(c),bias=True)
if $re != 0:
        for i in range(0,$re):
                if os.path.exists(f"$workdir/saved models/saved_model[{i+1}].pth"):
                                        model.load_state_dict(torch.load(f"$workdir/saved models/saved_model[{i+1}].pth"))
                                        print("Loaded model: $workdir/saved models/saved_model["+ str(i+1) + "].pth")
if torch.cuda.is_available():
    model.cuda()
print("--> model defined for use")
print(model)
################################
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())
print(f"optimizer used: {optimizer}")
epochs = $epochs - $re
if $re == 0:
    if os.path.exists(f"$workdir/saved models"):
        shutil.rmtree(f"$workdir/saved models")
    os.mkdir(f"$workdir/saved models")
torch.save(model.state_dict(), f"$workdir/saved models/saved_model[{$re}].pth")
min_valid_loss = np.inf
tl = np.array([])
ta = np.array([])
vl = np.array([])
va = np.array([])
print("--> beginning training")
for e in range(epochs):
    st = time.time()
    train_loss = 0.0
    train_acc = 0.0
    predict = []
    p = []
    model.train()     # Optional when not using Model Specific layer
    for data, labels in train_dataloader:
        if torch.cuda.is_available():
            data, labels = data.cuda(), labels.cuda()
        
        optimizer.zero_grad()
        target = model(data.float())
        t = torch.Tensor.cpu(target)
        l = torch.Tensor.cpu(labels)
        pred = np.argmax(t.detach().numpy(), axis=-1)
        predict = np.append(predict, pred)
        p = np.append(p, l.detach().numpy())

        loss = loss_fn(target,labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    train_acc += (predict == p).sum()


    #np.savetxt(f"$workdir/cm-target-C{len(c)}-D$size-{e+$re}.csv", t.detach().numpy(), delimiter=",")
    #np.savetxt(f"$workdir/cm-tlabels-C{len(c)}-D$size-{e+$re}.csv", l.detach().numpy(), delimiter=",")
    print("--> computing confusion matrix")
    cm = confusion_matrix(predict, p)
    print(cm)
    print("--> saving confusion matrix")
    np.savetxt(f"$workdir/cm-train-C{len(c)}-D$size-{e+$re}.csv", cm, delimiter=",")
    
    valid_loss = 0.0
    valid_acc = 0.0
    vpredict = []
    vp = []
    model.eval()     # Optional when not using Model Specific layer
    for vdata, vlabels in validation_dataloader:
        if torch.cuda.is_available():
            vdata, vlabels = vdata.cuda(), vlabels.cuda()

        
        vtarget = model(vdata.float())
        
        vt = torch.Tensor.cpu(vtarget)
        vl = torch.Tensor.cpu(vlabels)
        vpred = np.argmax(vt.detach().numpy(), axis=-1)
        vpredict = np.append(vpredict, vpred)
        vp = np.append(vp, vl.detach().numpy())
     
        vloss = loss_fn(vtarget,vlabels)
        valid_loss = vloss.item()*vdata.size(0)
    valid_acc += (vpredict == vp).sum()

    #np.savetxt(f"$workdir/cm-vtarget-C{len(c)}-D$size-{e+$re}.csv", vt.detach().numpy(), delimiter=",")
    #np.savetxt(f"$workdir/cm-vlabels-C{len(c)}-D$size-{e+$re}.csv", vl.detach().numpy(), delimiter=",")
    print("--> computing confusion matrix")
    vcm = confusion_matrix(vpredict, vp)
    print(vcm)
    print("--> saving confusion matrix")
    np.savetxt(f"$workdir/cm-valid-C{len(c)}-D$size-{e+$re}.csv", vcm, delimiter=",")
    et = time.time()
    rt = et-st

    print("--> testing model against test data (to see model accuracy)")
    tpredict = []
    tp = []
    model.eval()
    for i in range(0,int(ndata*0.05*len(c))):
        x, y = test_data[i][0], test_data[i][1]
        x = x.reshape(1,4,$imgsize,$imgsize)
        x = torch.from_numpy(x)
        x = x.float()
        with torch.no_grad():
            tpred = model (x.cuda()) if torch.cuda.is_available() else model(x)
            tpredicted, tactual = tpred[0].argmax(0), y 
            tpredicted = torch.Tensor.cpu(tpredicted)
            tpredict = np.append(tpredict, tpredicted)
            tp = np.append(tp, tactual)


    print("--> computing confusion matrix")
    tcm = confusion_matrix(tpredict, tp)
    print(tcm)
    print("--> saving confusion matrix")
    np.savetxt(f"$workdir/cm-test-C{len(c)}-D$size-{e+$re}.csv", tcm, delimiter=",")

    print(f'Epoch {e+1+$re} \t Runtime: {round(rt,2)}s \t Training Loss: {train_loss / len(train_dataloader)} \t Validation Loss: {valid_loss / len(validation_dataloader)}')
    # if min_valid_loss > valid_loss:
        # print(f'Validation Loss Decreased({min_valid_loss:.6f}--->{valid_loss:.6f}) \t Saving The Model')
        ##min_valid_loss = valid_loss
        # Saving State Dict
    torch.save(model.state_dict(), f"$workdir/saved models/saved_model[{e+1+$re}].pth")
    # else:
        # print(" ")
    
    tl = train_loss / len(train_dataloader)
    ta = train_acc / len(training_data)
    vl = valid_loss / len(validation_dataloader)
    va = valid_acc / len(validation_data)
   
    f = open("$workdir/tl.txt", "a+")
    f.write(str(tl) + "\n")
    print(f"--> stored training loss values: {tl}")
    f.close()

    f = open("$workdir/ta.txt", "a+")
    f.write(str(ta) + "\n")
    print(f"--> stored training accuracy values: {ta}")
    f.close()

    f = open("$workdir/vl.txt", "a+")
    f.write(str(vl) + "\n")
    print(f"--> stored validation loss values: {vl}")
    f.close()

    f = open("$workdir/va.txt", "a+")
    f.write(str(va) + "\n")
    print(f"--> stored validation accuracy values: {va}")
    f.close()


print("--> task complete")
EOD


cat > ${job} << EOD
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=42
#SBATCH --mem-per-cpu=3850
#SBATCH --gres=gpu:ampere_a100:1
#SBATCH --partition=gpu
#SBATCH --time=48:00:00
#SBATCH --account=su007-rr-gpu
module purge


# the following modules have been saved into collection PT
module load GCCcore/10.2.0
module load Python/3.8.6
module load GCC/10.2.0  CUDA/11.1.1  OpenMPI/4.0.5
module load PyTorch/1.9.0
module load torchvision/0.10.0-PyTorch-1.9.0 
module load scikit-learn/0.23.2 
module load matplotlib/3.3.3

#module restore PT
module list

srun $py
EOD

chmod 755 ${job}
chmod g+w ${job}
chmod 755 ${py}

sbatch ${job}
