import torchvision
import torch
import torch.nn as nn
import torchvision
from torchsummary import summary
from torchvision import datasets, models
import matplotlib.pyplot as plt
import sklearn
import numpy as np
import time
import random
import sys
sys.path.insert(0, '/home/physics/phsht/Projects/ML-Anderson3D/DataBased/PyTorch/3D/scripts-DB')
from AM_MLtools import *
import os
import copy
import re
##############################################################################
print(sys.argv)
    
if ( len(sys.argv) >=10):
    #SEED = 101
    SEED = int(sys.argv[1])
    my_size= int(sys.argv[2])
    my_size_samp=int(sys.argv[3])
    my_validation_split=0.1# float(sys.argv[4])
    my_batch_size=int(sys.argv[5])
    my_num_epochs= int(sys.argv[6])
    flag=int(sys.argv[7])
    mylr=float(sys.argv[8])
    psi_type=str(sys.argv[9])
    my_classes=[str(classe_ele) for classe_ele in sys.argv[10].split(',')]
    
else:
    print ('Number of', len(sys.argv), \
           'arguments is less than expected (10) --- ABORTING!')

print('--> defining parameters')
    
myseed=SEED
width= my_size
size_samp=my_size_samp
size_data=500
validation_split= my_validation_split
batch_size=my_batch_size
num_epochs= my_num_epochs
subclasses=my_classes
nb_classes=len(subclasses)
size_samp=my_size_samp*nb_classes
print('CLASSES',my_classes)
print('###############')
print(subclasses)
##############################################################################
dataname_og='L'+str(width)+'-'+str(size_data)+'-Ohtsuki'
dataname='L'+str(width)+'-'+str(my_size_samp)+'-Ohtsuki'
data_test='L'+str(width)+'-500-Ohtsuki-test'
#datapath = '/storage/disqs/'+'ML-Anderson3D/EvecRaws/'+dataname # SC-RTP
datapath = '/home/physics/phsht/Projects/ML-Anderson3D/Data/EvecPKL/'+dataname_og
#testpath = '/home/physics/phsht/Projects/ML-Anderson3D/Data/EvecPKL/'+data_test
print(os.listdir(datapath))
print(dataname,"\n",datapath)
##############################################################################
method='PyTorch-Ohtsuki-'+str(myseed)+'-e'+str(num_epochs)+'-bs'+str(batch_size)
modelname = 'Model_'+method+'_'+dataname+'.pth'
historyname = 'History_'+method+'_'+dataname+'.pkl'
print(method,"\n",modelname,"\n",historyname)

savepath = './'+dataname+'_Adam_'+str(batch_size)+'/'

try:
    os.mkdir(savepath)
except FileExistsError:
    pass

modelpath = savepath+modelname
historypath = savepath+historyname
cm_val_path=savepath+method+'_'+dataname+'cm_val_best.txt'
cm_test_path=savepath+method+'_'+dataname+'cm_test_best.txt'
print(savepath,modelpath,historypath)
#############################################################################################
class Ohtsuki3D(nn.Module):
    def __init__(self):
        super(Ohtsuki3D, self).__init__()

        self.Conv1 = nn.Conv3d(1, 64,5, stride=1,padding='valid',bias=False)
        self.Conv2 = nn.Conv3d(64,64,5, stride=1,padding='same',bias=False)
        self.maxpool1 = nn.MaxPool3d(kernel_size = 2, stride = 2)
        self.dropout1=nn.Dropout(p=0.5)

        self.Conv3 = nn.Conv3d(64, 96,3, stride=1,padding='valid',bias=False)
        self.Conv4 = nn.Conv3d(96,96,3, stride=1,padding='same',bias=False)
        self.maxpool2 = nn.MaxPool3d(kernel_size = 2, stride = 2)
        self.dropout2=nn.Dropout(p=0.5)

        self.Conv5 = nn.Conv3d(96,128,3, stride=1,padding='valid',bias=False)
        self.Conv6 = nn.Conv3d(128,128,3, stride=1,padding='same',bias=False)
        self.maxpool3 = nn.MaxPool3d(kernel_size = 2, stride = 2)
        self.dropout3=nn.Dropout(p=0.5)

        self.FC1 = nn.Linear(128*3*3*3, 1024,bias=False)
        self.dropout4=nn.Dropout(p=0.5)
        self.FC2 = nn.Linear(1024, 2,bias=False)

   
    def forward(self, x):
        out = F.relu(self.Conv1(x))
        out = F.relu(self.Conv2(out))
        out = self.maxpool1(out)
        out = self.dropout1(out)
        out = F.relu(self.Conv3(out))
        out = F.relu(self.Conv4(out))
        out = self.maxpool2(out)
        out = self.dropout2(out)
        out = F.relu(self.Conv5(out))
        out = F.relu(self.Conv6(out))
        out = self.maxpool3(out)
        out = self.dropout3(out)
        out = out.reshape(out.size(0), -1)
        out =self.FC1(out)
        out = self.dropout4(out)
        out =self.FC2(out)
        
        return out
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

test_subclasses=my_classes
train_subclasses=my_classes
print('train classes', train_subclasses)
print('test classes', test_subclasses)
print(os.getcwd())

print('--> reading CSV data')
if psi_type=='squared':
    temp_whole_dataset=MyDatasetFolder(root=datapath,loader=pkl_file_loader,transform=torchvision.transforms.ToTensor(),extensions='.pkl',subclasses=subclasses)
    #test_dataset=MyDatasetFolder(root=testpath,loader=pkl_file_loader,transform=torchvision.transforms.ToTensor(),extensions='.pkl',subclasses=test_subclasses)
else:
    temp_whole_dataset=MyDatasetFolder(root=datapath,loader=pkl_file_loader_psi,transform=torchvision.transforms.ToTensor(),extensions='.pkl',subclasses=subclasses)
    #test_dataset=MyDatasetFolder(root=testpath,loader=pkl_file_loader_psi,transform=torchvision.transforms.ToTensor(),extensions='.pkl',subclasses=test_subclasses)

if size_samp!=5000:
    print(size_samp)
    indices = torch.randperm(len(temp_whole_dataset))[:size_samp]
    whole_dataset = torch.utils.data.Subset(temp_whole_dataset,indices)
else:
    whole_dataset = whole_dataset
print('--> defining/reading DATA')
test_set=0
test_reject=0
os.getcwd()

data_size = len(whole_dataset)
#test_size = len(test_dataset)
# validation_split=0.1
split=int(np.floor(validation_split*data_size))
training=int(data_size-split)
#split_test=int(np.floor(0.5*test_size))
#size_test=int(test_size-split_test)
# split the data into training and validation
training_set, validation_set= torch.utils.data.random_split(whole_dataset,(training,split))
test_set, test_reject=torch.utils.data.random_split(test_dataset,(size_test,split_test))
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
#test = torch.utils.data.DataLoader(
#        dataset=test_set,
#        batch_size=batch_size,
#        num_workers=16,
#        shuffle=False)
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
model=Ohtsuki3D()

 #the model is sent to the GPU
model = model.to(device)
print('--> defining optimizer')
optimizer=torch.optim.Adam(model.parameters(),lr=mylr)
#optimizer=torch.optim.Adadelta(model.parameters(), lr=mylr, rho=0.9, eps=1e-06, weight_decay=0)
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
        model,train,val,test,
        device, 
        criterion,optimizer,
        num_epochs,exp_lr_scheduler,savepath, 
        method,dataname,modelname,modelpath,
        batch_size,class_names)
else:
    print('--> loading saved model')
    checkpoint=torch.load(modelpath+'_best.pth')
    model.load_state_dict(checkpoint['model_state_dict'],map_location='cpu')
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    val_loss=checkpoint['val loss']
    accuracy=checkpoint['train acc']
    _loss=checkpoint['train loss']
    val_accuracy=checkpoint['val acc']
    val_loss=checkpoint['val loss']
    epochs=checkpoint['train epoch']
    model.eval()

print('--> computing/saving confusion matrices')
#cm=simple_confusion_matrix(model,val,device,number_classes,class_names)
#np.savetxt(cm_path,cm,fmt='%d')
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
fig.savefig(savepath+method+'_'+dataname+'_loss'+'.png',transparent=True)

fig=plt.figure()
plt.plot(epochs,val_accuracy, label='val accuracy')
plt.plot(epochs,accuracy, label='training accuracy')
plt.legend(loc='upper left')
fig.savefig(savepath+method+'_'+dataname+'_accuracy'+'.png',transparent=True)

cm_val=simple_confusion_matrix(model,val,device,number_classes,class_names)
np.savetxt(cm_val_path,cm_val,fmt='%d')
cm_test=simple_confusion_matrix(model,test,device,number_classes,class_names)
np.savetxt(cm_test_path,cm_test,fmt='%d')
percentage_correct(model,device,class_names,test,savepath,method,dataname)


























