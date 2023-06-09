import torchvision
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import sklearn
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
import numpy as np
import time
import random
import seaborn as sns
import sys
sys.path.insert(0, '/home/p/phrhmb/Anderson/AM_code')
from AM_MLtools import *
from tqdm import tqdm, trange
import os
import copy
import re
##############################################################################
print(sys.argv)
    
if ( len(sys.argv) >= 9):
    #SEED = 101
    SEED = int(sys.argv[1])
    my_size= int(sys.argv[2])
    my_img_size= int(sys.argv[3])
    my_size_samp=int(sys.argv[4])
    my_validation_split= float(sys.argv[5])
    my_batch_size=int(sys.argv[6])
    my_num_epochs= int(sys.argv[7])
    flag=int(sys.argv[8])
    my_classes=['W15.0','W18.0']
else:
    print ('Number of', len(sys.argv), \
           'arguments is less than expected (2) --- ABORTING!')

print('--> defining parameters')
    
myseed=SEED
width= my_size
size_samp=my_size_samp
img_sizeX= my_img_size
validation_split= my_validation_split
batch_size=my_batch_size
num_epochs= my_num_epochs
subclasses=my_classes
nb_classes=len(subclasses)
print('CLASSES',my_classes)
print('###############')
print(subclasses)
dataname='L'+str(width)+'-'+str(size_samp)+'-s'+str(img_sizeX)+'-'+str(nb_classes)+'-classes'
data_dir='L'+str(width)+'-'+str(size_samp)+'-s'+str(img_sizeX)
#datapath = '/storage/disqs/'+'ML-Anderson3D/EvecRaws/'+dataname # SC-RTP
datapath = '/home/p/phrhmb/Anderson/Data/images/'+data_dir
#print(os.listdir(datapath))
print(dataname,"\n",datapath)

method='PyTorch-resnet18-'+str(myseed)+'-e'+str(num_epochs)+'-bs'+str(batch_size)
modelname = 'Model_'+method+'_'+dataname+'.pth'
historyname = 'History_'+method+'_'+dataname+'.pkl'
print(method,"\n",modelname,"\n",historyname)

savepath = './'+dataname+'_Adam_1-4_'+str(batch_size)+'/'

try:
    os.mkdir(savepath)
except FileExistsError:
    pass

modelpath = savepath+modelname
historypath = savepath+historyname
cm_path=savepath+method+'_'+dataname+'cm_val_best.txt'
print(savepath,modelpath,historypath)
#############################################################################################

print('--> defining seeds')
torch.manual_seed(myseed+1)
np.random.seed(myseed+2)
torch.cuda.manual_seed(myseed+3)
#torch.cuda.seed()
#torch.cuda.seed_all()
random.seed(myseed+4)

print('--> defining ML lib versions and devices')
print('torch version:',torch.__version__)
print('torchvision version:',torchvision.__version__)
print('sklearn version:', sklearn.__version__)
t=torch.Tensor()
print('current device: ', t.device, t.dtype, t.layout)

# switch to GPU if available
device=t.device
if torch.cuda.is_available():
    device = torch.device("cuda:0")
print('chosen device: ',device)

print(os.getcwd())

print('--> reading CSV data')
whole_dataset=MyImageFolder(root=datapath,loader = torchvision.datasets.folder.default_loader,extensions='.png',subclasses=subclasses,transform=torchvision.transforms.ToTensor())

print('--> defining/reading DATA')
training_set=0
validation_set=0
os.getcwd()

data_size = len(whole_dataset)
# validation_split=0.1
split=int(np.floor(validation_split*data_size))
training=int(data_size-split)
# split the data into training and validation
training_set, validation_set= torch.utils.data.random_split(whole_dataset,(training,split))

print('--> loading training data')
train = torch.utils.data.DataLoader(
        dataset=training_set,
        batch_size=batch_size,
        num_workers=16,
        shuffle=True)

print('--> loading validation data')
val = torch.utils.data.DataLoader(
        dataset=validation_set,
        batch_size=batch_size,
        num_workers=16,
        shuffle=False)

print('--> defining classes/labels')
class_names = whole_dataset.classes
print(class_names)
inputs,labels,paths= next(iter(train))

img_sizeX,img_sizeY= inputs.shape[-1],inputs.shape[-2]
num_of_train_samples = len(training_set) # total training samples
num_of_test_samples = len(validation_set) #total validation samples
steps_per_epoch = np.ceil(num_of_train_samples // batch_size)
number_classes = len(class_names)

print('--> protocolling set-up')
print('number of samples in the training set:', num_of_train_samples)
print('number of samples in the validation set:', num_of_test_samples )
print('number of samples in a training batch',len(train)) 
print('number of samples in a validation batch',len(val))
print('number of classes',number_classes )

# ## building the CNN
print('--> building the CNN')
model=models.resnet18(pretrained=True, progress=True)
num_ftrs = model.fc.in_features # number of input features of the last layer which is fully connected (fc)

#We modify the last layer in order to have 2 output: percolating or not
model.fc=nn.Linear(num_ftrs, number_classes )
 #the model is sent to the GPU
model = model.to(device)

print('--> defining optimizer')
optimizer=torch.optim.Adam(model.parameters(),lr=0.0001)
#optimizer=torch.optim.Adadelta(model.parameters(), lr=1.0, rho=0.9, eps=1e-06, weight_decay=0)
# defining the loss function
criterion = nn.CrossEntropyLoss()

exp_lr_scheduler= torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

# checking if GPU is available
if torch.cuda.is_available():
    model = model.cuda()
    criterion = criterion.cuda()

#the model is sent to the GPU
#model=model.to(device)
model=model.double()
if flag==0:
    print('--> starting training epochs')
    print('number of epochs',num_epochs )
    print('batch_size',batch_size )
    print('number of classes',number_classes )

    base_model = train_model(
        model,train,val,
        device, 
        criterion,optimizer,
        num_epochs,exp_lr_scheduler,savepath, 
        method,dataname,modelname,modelpath,
        batch_size,class_names)
else:
    print('--> loading saved model')
    checkpoint=torch.load(modelpath+'.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    val_loss=checkpoint['val loss']
    accuracy=checkpoint['train acc']
    _loss=checkpoint['train loss']
    val_accuracy=checkpoint['val acc']
    val_loss=checkpoint['val loss']
    epochs=checkpoint['train epoch']
    model.eval()

print('--> computing/saving confusion matrices')
loss_file=[file for file in os.listdir(savepath) if 'loss.txt' in file]
data=np.loadtxt(savepath+'/'+loss_file[0],unpack=True)
epochs=data[0]
_loss=data[1]
accuracy=data[2]
val_loss=data[3]
val_accuracy=data[4]

fig=plt.figure()
plt.plot(epochs,val_loss, label='val loss')
plt.plot(epochs,_loss, label='training loss')
plt.legend(loc='upper left')
fig.savefig(savepath+method+'_'+dataname+'_loss'+'.png')

fig=plt.figure()
plt.plot(epochs,val_accuracy, label='val accuracy')
plt.plot(epochs,accuracy, label='training accuracy')
plt.legend(loc='upper left')
fig.savefig(savepath+method+'_'+dataname+'_accuracy'+'.png')

cm=simple_confusion_matrix(model,val,device,number_classes,class_names)
np.savetxt(cm_path,cm,fmt='%d')

percentage_correct(model,device,class_names,val,savepath,method,dataname)












































