import torchvision
import torch.nn.functional as F
import torch
import torch.nn as nn
from typing import Any, Callable, cast, Dict, List, Optional, Tuple, Union
import torch.optim as optim
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
import numpy as np
from PIL import Image
import time
import pandas as pd
import csv
import seaborn as sns
from tqdm import tqdm, trange
import os
import copy

EXTENSIONS = ('.raw', '.pkl', '.txt')
class MyDatasetFolder(torchvision.datasets.DatasetFolder):
    # override the __getitem__ method. this is the method that dataloader calls
    def __init__(
        self,
        root,
        loader,
        subclasses=[],
        extensions=None,
        transform=torchvision.transforms.ToTensor(),
        target_transform=None,
        is_valid_file=None):
        self.subclasses=subclasses
        super().__init__(root,loader,EXTENSIONS if is_valid_file is None else None,transform=transform, 
                         target_transform=target_transform,is_valid_file=is_valid_file)
        
        classes, class_to_idx, old_classes, old_class_to_idx= self.find_new_classes(self.root)
        samples = self.make_new_dataset(self.root,old_class_to_idx, class_to_idx,extensions, is_valid_file)
        self.subclasses=subclasses
        print('SUBCLASSES',self.subclasses)
        self.loader = loader
        self.extensions = extensions
        self.old_classes = old_classes
        self.old_class_to_idx =old_class_to_idx
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples
        self.targets = [s[1] for s in samples]
    def has_file_allowed_extension(self,filename, extensions):
        return filename.lower().endswith(extensions if isinstance(extensions, str) else tuple(extensions))

    def make_new_dataset(self,directory,
    old_class_to_idx,
    class_to_idx,
    extensions,
    is_valid_file):
        directory = os.path.expanduser(directory)

        if old_class_to_idx or class_to_idx is None:
            _,class_to_idx, __,old_class_to_idx = self.find_new_classes(directory)
        elif notold_class_to_idx:
            raise ValueError("'class_to_index' must have at least one entry to collect any samples.")
        elif not class_to_idx:
            raise ValueError("'new_class_to_index' must have at least one entry to collect any samples.")

        both_none = extensions is None and is_valid_file is None
        both_something = extensions is not None and is_valid_file is not None
        if both_none or both_something:
            raise ValueError("Both extensions and is_valid_file cannot be None or not None at the same time")

        if extensions is not None:

            def is_valid_file(x: str) -> bool:
                return self.has_file_allowed_extension(x, extensions)  # type: ignore[arg-type]

        is_valid_file = cast(Callable[[str], bool], is_valid_file)

        instances = []
        available_classes = set()
        for target_class, old_target_class in zip(sorted(class_to_idx.keys()),sorted(old_class_to_idx.keys())):
            print('target',old_target_class,'new_target', target_class)
            old_class_index = old_class_to_idx[old_target_class]
            class_index = class_to_idx[target_class]
            old_target_dir = os.path.join(directory, old_target_class)
            if not os.path.isdir(old_target_dir):
                continue
            for root, _, fnames in sorted(os.walk(old_target_dir, followlinks=True)):
                for fname in sorted(fnames):
                    path = os.path.join(root, fname)
                    if is_valid_file(path):
                        item = path, class_index, old_class_index
                        instances.append(item)

                        if old_target_class not in available_classes:
                            available_classes.add(old_target_class)

        empty_classes = set(old_class_to_idx.keys()) - available_classes
        if empty_classes:
            msg = f"Found no valid file for the classes {', '.join(sorted(empty_classes))}. "
            if extensions is not None:
                msg += f"Supported extensions are: {extensions if isinstance(extensions, str) else ', '.join(extensions)}"
            raise FileNotFoundError(msg)

        return instances
    def find_new_classes(self,directory):
        print('type root', directory)
        old_classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())
        old_classes=[entry for entry in old_classes if entry in self.subclasses]
        prefix='W'
        size=self.root.split('-')[0].split('L')[1]
        print(size)
        if any('A3' in ele for ele in old_classes)==True:
            classes=[prefix+ele.split('hD')[1] for ele in old_classes]
        else:
            classes=old_classes        
        if not old_classes:
            raise FileNotFoundError(f"Couldn't find any class folder in {directory}.")
        
        old_class_to_idx = {cls_name: i for i, cls_name in enumerate(old_classes)}
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        
        return classes,class_to_idx, old_classes, old_class_to_idx
    def __getitem__(self, index):
        path, target,__ = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
       #print('tuple', self.class_to_idx)
        return sample, target, path

######################################################################
class former_MyDatasetFolder(torchvision.datasets.DatasetFolder):
    # override the __getitem__ method. this is the method that dataloader calls
    def __init__(
        self,
        root,
        loader,
        subclasses=[],
        extensions=None,
        transform=torchvision.transforms.ToTensor(),
        target_transform=None,
        is_valid_file=None):
        self.subclasses=subclasses
        super().__init__(root,loader,EXTENSIONS if is_valid_file is None else None,transform=transform, 
                         target_transform=target_transform,is_valid_file=is_valid_file)
        
        
        classes, class_to_idx = self.find_classes(self.root)
        samples = self.make_dataset(self.root, class_to_idx, extensions, is_valid_file)

        self.loader = loader
        self.extensions = extensions
        
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples
        self.targets = [s[1] for s in samples]
        #print(self.subclasses)

    def find_classes(self,directory):
        classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())
        classes=[entry for entry in classes if entry in self.subclasses]
        #print(classes)
        if not classes:
            raise FileNotFoundError(f"Couldn't find any class folder in {directory}.")
    
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx

    def __getitem__(self, index):
        # this is what DatasetFolder normally returns 
        original_tuple = super(MyDatasetFolder, self).__getitem__(index)
        # the image file path
        path = self.samples[index][0]
        # make a new tuple that includes original and the path
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path
#####################################################################################
class reg_Dataset_csv(torch.utils.data.Dataset):
    def __init__(self, csv_file, root_dir, loader,transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.csv_data = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.csv_file = csv_file
        name_column='disorder'
        classes=[]
        classe_order=[]
        self.name_column=name_column

        with open(csv_file) as f:
            cf = csv.reader(f)
            for column in cf:
                classes.append(column[1])
        classes=classes[1:]

        df = pd.DataFrame(classes, columns=['classes'])
        classes_ordered=df.drop_duplicates(subset=['classes'], keep='first')
        classes_unprocessed=classes_ordered['classes'].tolist()
        classes = [float(num) for num in classes_unprocessed]
        #print(classes)
        classes.sort()
        self.classes = classes
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        self.class_to_idx=class_to_idx


    def _find_classes(self,root_dir,csv_file):
        classes=[]
        with open(csv_file) as f:
            cf = csv.reader(f)
            for column in cf:
                classes.append(column[1])
        classes=classes[1:]
        df = pd.DataFrame(classes, columns=['classes'])
        classes_ordered=df.drop_duplicates(subset=['classes'], keep='first')
        classes_unprocessed=classes_ordered['classes'].tolist()
        classes = [float(num) for num in classes_unprocessed]
        #print('ici',classes)
        classes.sort()
        self.classes = classes
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        self.class_to_idx=class_to_idx
        return classes, class_to_idx
    def __len__(self):
        return len(self.csv_data)

    def __getitem__(self, idx):
        global class_to_idx
        if torch.is_tensor(idx):
            idx = idx.tolist()
        data_idx=self.csv_data.iloc[idx, 0]
        path = os.path.join(self.root_dir,
                                data_idx)
        #print(path)
        #print('######')
        data=pkl_file_loader(path)
        classes, class_to_idx=self._find_classes(self.root_dir,self.csv_file)
        label=float(self.csv_data.iloc[idx, 1])

        sample = {'data': data, 'labels': label, 'path':path}

        if self.transform:
            sample['data'] = self.transform(sample['data'])
        data= sample['data']
        labels=sample['labels']
        paths=sample['path']
        #print(time.time()-start)
        return data, labels,paths

#####################################################################################
#class MyImageFolder(torchvision.datasets.ImageFolder):
#    """Custom dataset that includes image file paths. Extends
#    torchvision.datasets.ImageFolder
#    """
##
#    # override the __getitem__ method. this is the method that dataloader calls
#    def __getitem__(self, index):
#        # this is what ImageFolder normally returns 
#        original_tuple = super(MyImageFolder, self).__getitem__(index)
#        # the image file path
#        path = self.imgs[index][0]
#        # make a new tuple that includes original and the path
##        tuple_with_path = (original_tuple + (path,))
#        return tuple_with_path

IMG_EXTENSIONS = ('.png', '.jpeg', '.svg')
class MyImageFolder(torchvision.datasets.DatasetFolder):
    def __init__(
        self,
        root,
        loader,
        subclasses=[],
        extensions=None,
        transform=None,
        target_transform=None,
        is_valid_file=None):
        self.subclasses=subclasses
        super().__init__(root,loader,IMG_EXTENSIONS if is_valid_file is None else None,transform=transform, 
                         target_transform=target_transform,is_valid_file=is_valid_file)
        
        
        
        classes, class_to_idx = self.find_classes(self.root)
        samples = self.make_dataset(self.root, class_to_idx, extensions, is_valid_file)

        self.loader = loader
        self.extensions = extensions
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples
        self.targets = [s[1] for s in samples]
        print(self.subclasses)

    def find_classes(self,directory):
        classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())
        classes=[entry for entry in classes if entry in self.subclasses]
        #print(classes)
        if not classes:
            raise FileNotFoundError(f"Couldn't find any class folder in {directory}.")
    
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx

    def __getitem__(self, index):
        # this is what ImageFolder normally returns 
        original_tuple = super(MyImageFolder, self).__getitem__(index)
        # the image file path
        path = self.samples[index][0]
        # make a new tuple that includes original and the path
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path
############################################################################
def pil_loader(input):
    with open(input, "rb") as f:
        img = Image.open(f)
        img.load()
    return img.convert("RGBA")
#####################################################################################
def raw_file_loader(input):
    #print(input)
    functions = np.loadtxt(input, comments="#", delimiter="\n", unpack=False)
    size=int(((input.split('/')[-1]).split('-')[0]).split('Evec0')[1])
    functions=functions.reshape(size,size,size)
    functions=functions*functions
    return functions

#####################################################################################
def pkl_file_loader(input):
    import pickle as pkl
    from numpy import loadtxt
    functions = pkl.load(open(input, 'rb'))
    if 'Evec0' in input:
        size=int(((input.split('/')[-1]).split('-')[0]).split('Evec0')[1])
    else:
        size=int(((input.split('/')[-1]).split('-')[2]).split('Evec-A3-M0')[0].replace('M',''))
    functions=functions.reshape(size,size,size)
    functions=functions**2
    return functions
#####################################################################################
def pkl_file_loader_abs(input):
    import pickle as pkl
    from numpy import loadtxt
    functions = pkl.load(open(input, 'rb'))
    if 'Evec0' in input:
        size=int(((input.split('/')[-1]).split('-')[0]).split('Evec0')[1])
    else:
        size=int(((input.split('/')[-1]).split('-')[2]).split('Evec-A3-M0')[0].replace('M',''))
    functions=functions.reshape(size,size,size)
    functions=abs(functions)**2
    return functions
#####################################################################################
def pkl_file_loader_psi(input):
    import pickle as pkl
    from numpy import loadtxt
    functions = pkl.load(open(input, 'rb'))
    if 'Evec0' in input:
        size=int(((input.split('/')[-1]).split('-')[0]).split('Evec0')[1])
    else:
        size=int(((input.split('/')[-1]).split('-')[2]).split('Evec-A3-M0')[0].replace('M',''))
    functions=functions.reshape(size,size,size)
    return functions


#####################################################################################
def train_model(model,train,val,device,criterion, optimizer, num_epochs, scheduler,savepath, method,dataname,modelname,modelpath,batch_size,class_names):
    print('ok')
    start_epoch=0
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    accuracy=[]
    _loss=[]
    val_accuracy=[]
    val_loss=[]
    epochs=[]
    val_epochs=[]
    number_classes=len(class_names)
    temp_class_names=class_names
    str_classes_names=" ".join(str(x) for x in temp_class_names)
    temp_list_model=[files for files in os.listdir(savepath) if files.startswith(modelname) and files.endswith('.pth') ]
    print('#############################')
    print('modelpath',modelname)
    first_parameter = next(model.parameters())
    input_length = len(first_parameter.size())
    if len(temp_list_model)!=0:
        list_model=[savepath + files for files in temp_list_model]
        print(list_model[0])
        print(os.getcwd())
        list_model.sort(key=os.path.getctime)
        print(list_model)
        checkpoint=torch.load(list_model[-1])

        model.load_state_dict(checkpoint['model_state_dict'])

        model.train()
        best_model_wts = copy.deepcopy(model.state_dict())
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        val_loss=checkpoint['val loss']
        accuracy=checkpoint['train acc']
        _loss=checkpoint['train loss']
        val_accuracy=checkpoint['val acc']
        val_loss=checkpoint['val loss']
        epochs=checkpoint['train epoch']
        best_loss=min(val_loss)
        start_epoch=max(epochs)+1
        print('Checkpoint found, training restarted at epoch: '+str(start_epoch))
    since=time.time()
    init=time.time()
    for epoch in range(start_epoch,num_epochs):
        print('Epoch {}/{}'.format(epoch+1,num_epochs))
        print('-' * 10)

    #two phases training and validating
        for phase in [train,val]:
            if phase == train:
                #print('Training', end=" ")
                len_dataset=len(train.dataset)
                name_phase='Training'
                model.train() # set the model to training mode
            else:
                #print('Validation', end=" ")
                len_dataset=len(val.dataset)
                name_phase='Validation'
                model.eval() # set the model to evaluation mode
            running_loss=0.0
            running_corrects=0.0
            for i, (inputs,labels,paths) in tqdm(enumerate(phase), total=int(len_dataset/phase.batch_size),desc=name_phase):
            
                if input_length> len(inputs.shape):
                    inputs=inputs.unsqueeze(1)
                inputs=inputs.double()
                inputs=inputs.to(device)
                labels=labels.to(device)

                #put the gradient to zero to avoid accumulation during back propagation
                optimizer.zero_grad()
                #now we need to carry out the forward and backward process in different steps
                #First the forward training
                #for the training step we need to log the loss
                with torch.set_grad_enabled(phase==train):
                    outputs=model(inputs)
                    outputs=outputs.double()
                    _, preds= torch.max(outputs,1)
                    loss=criterion(outputs,labels)
                #still for the training phase we need to implement backword process and optimization
                    if phase==train:
                        loss.backward()
                        optimizer.step()
                # We want variables to hold the loss statistics
                #loss.item() extract the loss value as float then it is multiply by the batch size
                running_loss+=loss.item()*inputs.size(0)
                running_corrects+= torch.sum(preds==labels.data).item()
            if phase == train:
                scheduler.step()

            if phase == train:
                epoch_loss= running_loss/len(phase.dataset)
                epoch_acc = running_corrects/ len(phase.dataset)
                print('{} loss= {:4f}, accuracy= {:4f}'.format(
                    'Training result:', epoch_loss, epoch_acc))
                accuracy.append(epoch_acc)
                _loss.append(epoch_loss)
                epochs.append(epoch)

            if phase == val:
                epoch_loss= running_loss/len(val.dataset)
                epoch_acc = running_corrects/len(val.dataset)
                print('{} val_loss= {:4f}, val_accuracy= {:4f}'.format(
                    'Validation result:', epoch_loss, epoch_acc))
                val_accuracy.append(epoch_acc)
                val_loss.append(epoch_loss)
                val_epochs.append(epoch)
            # Make a copy of the model if the accuracy on the validation set has improved
            if phase == val and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
        cm=simple_confusion_matrix(model,val,device,number_classes,class_names)
        np.savetxt(savepath+method+'_'+dataname+'cm_val_'+str(epoch)+'.txt',cm,fmt='%d',header=str_classes_names,comments='')
        cm_train=simple_confusion_matrix(model,train,device,number_classes,class_names)
        np.savetxt(savepath+method+'_'+dataname+'cm_train_'+str(epoch)+'.txt',cm_train,fmt='%d',header=str_classes_names,comments='')
        model.load_state_dict(best_model_wts)
        train_data=list(zip(epochs,_loss,accuracy,val_loss,val_accuracy))
        #print(train_data)
        header = '{0:^5s}   {1:^7s}   {2:^5s}   {3:^8s}   {4:^7s}'.format('epochs', 'loss', \
        'accuracy', 'val loss',   'val accuracy')
        filename=savepath+method+'_'+dataname+'accuracy_loss'+'.txt'
        np.savetxt(filename, train_data, header=header, fmt=['    %d  ','  %.7f','  %.7f','  %.7f','     %.7f'])
        if time.time()-init>18000:
            torch.save({'train epoch': epochs,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train acc': accuracy,
            'val acc' : val_accuracy,
            'train loss': _loss,
            'val loss' : val_loss,
            'cm':cm}, modelpath+'_epochs_'+str(epoch)+'.pth')
            init=time.time()
            print('saved')
    torch.save({'train epoch': epochs,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train acc': accuracy,
                'val acc' : val_accuracy,
                'train loss': _loss,
                'val loss' : val_loss,
                'cm':cm}, modelpath+'.pth')
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # Now we'll load in the best model weights and return it
    model.load_state_dict(best_model_wts)
    train_data=list(zip(epochs,_loss,accuracy,val_loss,val_accuracy))
    header = '{0:^5s}   {1:^7s}   {2:^5s}   {3:^8s}   {4:^7s}'.format('epochs', 'loss', \
    'accuracy', 'val loss',   'val accuracy')
    filename=savepath+method+'_'+dataname+'accuracy_loss'+'.txt'
    np.savetxt(filename, train_data, header=header, fmt=['    %d  ','  %.7f','  %.7f','  %.7f','     %.7f'])

    return model, accuracy, _loss, val_accuracy, val_loss, epochs, val_epochs
#####################################################################################

#####################################################################################
def train_reg_model(model,train,val,device,criterion, optimizer, num_epochs, scheduler,savepath, method,dataname,modelname,modelpath,batch_size,class_names):
    best_loss = 20000.0
    _loss=[]
    val_loss=[]
    epochs=[]
    val_epochs=[]
    temp_list_model=[files for files in os.listdir(savepath) if files.startswith(modelname) and files.endswith('.pth') ]
    print('#############################')
    print('modelpath',modelname)
    if len(temp_list_model)!=0:
        list_model=[savepath + files for files in temp_list_model]
        print(list_model[0])
        print(os.getcwd())
        list_model.sort(key=os.path.getctime)
        print(list_model)
        checkpoint=torch.load(list_model[-1])

        model.load_state_dict(checkpoint['model_state_dict'])

        model.train()
        best_model_wts = copy.deepcopy(model.state_dict())
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        val_loss=checkpoint['val loss']

        _loss=checkpoint['train loss']

        val_loss=checkpoint['val loss']
        epochs=checkpoint['train epoch']
        best_loss=min(val_loss)
        start_epoch=max(epochs)+1
        print('Checkpoint found, training restarted at epoch: '+str(start_epoch))
    since=time.time()
    init=time.time()
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1,num_epochs))
        print('-' * 10)

    #two phases training and validating
        for phase in [train,val]:
            if phase == train:
                #print('Training', end=" ")
                len_dataset=len(train.dataset)
                name_phase='Training'
                model.train() # set the model to training mode
            else:
                #print('Validation', end=" ")
                len_dataset=len(val.dataset)
                name_phase='Validation'
                model.eval() # set the model to evaluation model

            running_loss=0.0
            running_corrects=0.0

            # Here's where the training happens
            # print('--- iterating through data ...')

            for i, (inputs,labels,paths) in tqdm(enumerate(phase), total=int(len_dataset/phase.batch_size),desc=name_phase):
                inputs=inputs.double()      
                labels=labels.double()
                inputs=inputs.to(device)
                labels=labels.to(device)
                #print(labels[0],labels[1], labels[2], labels[3], labels[4])
                #print(paths[0],paths[1], paths[2], paths[3], paths[4])
                #labels = [_label.cuda() for _label in label]
                inputs=inputs.unsqueeze(1)
                labels = labels.unsqueeze(1)
                #paths=paths.to(device)
                #print(labels[0],labels[1], labels[2], labels[3], labels[4])
                #print(paths[0],paths[1], paths[2], paths[3], paths[4])
                #put the gradient to zero to avoid accumulation during back propagation
                optimizer.zero_grad()

                #now we need to carry out the forward and backward process in different steps
                #First the forward training
                #for the training step we need to log the loss
                with torch.set_grad_enabled(phase==train):
                    outputs=model(inputs)
                    loss=criterion(outputs,labels.double())

                #still for the training phase we need to implement backword process and optimization

                    if phase==train:
                        loss.backward()
                        optimizer.step()

                # We want variables to hold the loss statistics
                #loss.item() extract the loss value as float then it is multiply by the batch size
                running_loss+=loss.item()*inputs.size(0)


            if phase == train:
                scheduler.step()

            if phase == train:
                epoch_loss= running_loss/len(phase.dataset)
                print('{} loss= {:4f}'.format(
                    'Training result:', epoch_loss))
                _loss.append(epoch_loss)
                epochs.append(epoch)

            if phase == val:
                epoch_loss= running_loss/len(val.dataset)
                print('{} val_loss= {:4f}'.format(
                     'Validation result:', epoch_loss))
                val_loss.append(epoch_loss)
                val_epochs.append(epoch)

            # Make a copy of the model if the accuracy on the validation set has improved
            if phase == val and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())
        if time.time()-init>=32400:
            model.load_state_dict(best_model_wts)
            train_data=list(zip(epochs,_loss,val_loss))
            header = '{0:^5s}   {1:^7s}   {2:^7s}'.format('epochs', 'loss', \
             'val loss')
            filename=savepath+method+'_'+dataname+'accuracy_loss'+'.txt'
            np.savetxt(filename, train_data, header=header, fmt=['    %d  ','  %.7f','  %.7f'])

            torch.save({'train epoch': epochs,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train loss': _loss,
            'val loss' : val_loss }, modelpath+'_epochs_'+str(epoch)+'.pth')
            init=time.time()
            print('saved')

        print()
    epochs=[epochs[i]+1 for i in range(len(epochs))]

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val loss: {:4f}'.format(best_loss))

    # Now we'll load in the best model weights and return it
    model.load_state_dict(best_model_wts)
    train_data=list(zip(epochs,_loss,val_loss))
    header = '{0:^5s}   {1:^7s}     {2:^8s}   '.format('epochs', 'loss', \
     'val loss')
    filename=savepath+method+'_'+dataname+'accuracy_loss'+'.txt'
    np.savetxt(filename, train_data, header=header, fmt=['    %d  ','  %.7f','  %.7f'])
    torch.save({'train epoch': epochs,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train loss': _loss,
            'val loss' : val_loss }, modelpath+'_epochs_'+str(epoch)+'.pth')

    return model, _loss, val_loss, epochs, val_epochs
##################################################################################
class Resnet18_regression(nn.Module):
    def __init__(self,resnet18_model):
        super(Resnet18_regression, self).__init__()
        self.resnet18_model=resnet18_model
        num_ftrs = self.resnet18_model.fc.out_features
        self.new_layers=nn.Sequential(nn.Linear(num_ftrs,1))#nn.ReLU(),
                                      #nn.Linear(num_ftrs,256),
                                      #nn.ReLU(),
                                      #nn.Linear(256,64),
                                      #nn.Linear(64,1))

    def forward(self,x):
        x=self.resnet18_model(x)
        x=self.new_layers(x)
        return x

#####################################################################################
def visualize_model(model, num_images=6):
    was_training = model.training
    model.eval()
    images_handeled = 0
    fig = plt.figure()
    first_parameter = next(model.parameters())
    input_length = len(first_parameter.size())
    with torch.no_grad():
        for i, (inputs,labels,paths) in enumerate(val):
            if input_length> len(inputs.shape):
                inputs=inputs.unsqueeze(1)
            inputs=inputs.to(device)
            labels=labels.to(device)

            outputs = model(inputs) #value of the output neurons
            _, preds = torch.max(outputs, 1) #gives the max value and stores in preds the neurons to which it belongs

            for j in range(inputs.size()[0]):
                images_handeled += 1
                ax = plt.subplot(num_images//2, 2, images_handeled)
                ax.axis('off')
                ax.set_title('predicted: {}; \n true label {}; \n path: {};'.format(class_names[preds[j]] ,
                                                                     class_names[labels[j]],paths[j])
                            )
                imshow(inputs.cpu().data[j])
                
                if images_handeled == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)
###################################################################################
def visualize_model_misclassified(model, num_images=6): #gives shows only the misclassified images
    import re
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()
    first_parameter = next(model.parameters())
    input_length = len(first_parameter.size())
    with torch.no_grad():
        for i, (inputs,labels,paths) in enumerate(val):
            if input_length> len(inputs.shape):
                inputs=inputs.unsqueeze(1)
            inputs=inputs.to(device)
            labels=labels.to(device)
            
            outputs = model(inputs) #value of the output neurons
            _, preds = torch.max(outputs, 1) #gives the max value and stores in preds the neurons to which it belongs

            for j in range(inputs.size()[0]):
                if labels[j]!=preds[j] and abs(labels[j]-preds[j])>4:
                #print(inputs.size()[0])
                    images_so_far += 1
                    ax = plt.subplot(num_images//2, 2, images_so_far)
                    ax.axis('off')
                    ax.set_title('predicted: {}; \n true label {}; \n path: {};'.format(class_names[preds[j]] ,
                                                                     class_names[labels[j]],paths[j])
                            )
                    imshow(inputs.cpu().data[j])
                
                    if images_so_far == num_images:
                        model.train(mode=was_training)
                        return
        
        model.train(mode=was_training)
#######################################################################################

def simple_confusion_matrix(model,loader,device,number_classes,class_names):
    with torch.no_grad():
        confusion_matrix = torch.zeros(number_classes, number_classes)
        for i, (data) in enumerate(loader):
            inputs=data[0]
            labels=data[1]
            if inputs.shape[1]>4:
                inputs=inputs.unsqueeze(1)
            inputs=inputs.double()
            labels=labels.double()
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            for p,t in zip(preds.view(-1),labels.view(-1)):
		#confusion_matrix[t.long(), p.long()] += 1
                confusion_matrix[p.long(), t.long()] += 1
    return confusion_matrix

################################################################################
def confusion_matrix_torch(cm, target_names,cmap=None,title='Confusion Matrix'):
    import matplotlib.pyplot as plt
    import numpy as np
    import itertools
    import matplotlib.ticker as plticker



    #accuracy = np.trace(cm) / float(np.sum(cm))
    #misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    fig=plt.subplots(figsize=(20,20))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title,fontsize=30)
    cbar=plt.colorbar()
    cbar.ax.tick_params(labelsize=30) 

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45, fontsize=30)
        plt.yticks(tick_marks, target_names,fontsize=30)

    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j]),
                    ha="center", va="center",
                    color="white" if  cm[i, j] == 0 or cm[i, j] > thresh else "black") 


    plt.tight_layout()
    plt.rcParams.update({'font.size': 20})
    plt.ylim(len(cm)-0.5, -0.5)
    plt.ylabel('True label',fontsize=30)
    plt.xlabel('Predicted label',fontsize=30)
    plt.savefig(savepath+method+'_'+dataname+'_CM'+'.png')
    plt.show()
##################################################################################
def percentage_correct(model,device,class_names,val,savepath,method,dataname):
    number_classes=len(class_names)
    class_correct = list(0. for i in range(number_classes))
    class_total = list(0. for i in range(number_classes))
    accuracy=list(0. for i in range(number_classes))
    average=list(0. for i in range(number_classes))
    model=model.to(device)
    first_parameter = next(model.parameters())
    input_length = len(first_parameter.size())
    with torch.no_grad():
        for i, (data) in enumerate(val):
            inputs=data[0]
            labels=data[1]
            if input_length> len(inputs.shape):
                inputs=inputs.unsqueeze(1)
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1) 

            c = (preds == labels).squeeze()
            for i in range(inputs.size()[0]):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    for i in range(number_classes):
        average[i]=(class_correct[i] / class_total[i])*100
    
        print('Accuracy of %5s : %2d %%' % (
            class_names[i], 100 * class_correct[i] / class_total[i]))

    print(len(average))

    plt.figure(figsize=(14,14))
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.plot(class_names,average)
    plt.savefig(savepath+method+'_'+dataname+'_classacc'+'.png')    


def reg_prediction(dataloader,model,size,myseed,whole_dataset,savepath,nb_classes=17,data_type='test'):
    was_training = model.training
    model.eval()
    list_paths=[]
    list_labels=[]
    list_preds=[]
    list_p=[]
    model=model.to('cpu')
    header_l=['path','density','true label','prediction']

    with torch.no_grad():
        for i, (inputs,labels,paths) in enumerate(dataloader):
            inputs=inputs.to('cpu')
            labels=labels.to('cpu')
            labels=labels.unsqueeze(1)
            inputs=inputs.unsqueeze(1)
            #inputs=inputs.double()
            labels.numpy()
            predictions = model(inputs) #value of the output neurons

            for j in range(inputs.size()[0]):
                #p_occ=paths[j].split('_')[7]
                #regex2 = re.compile('\d+\.\d+')
                #p_reg=re.findall(regex2,p_occ)
                #print(p_reg)
                #p=float(p_reg[0])
               # paths_pred=[paths[j],p,labels[j].item(),pred[j].numpy()]
                temp_paths=paths[j]
                temp_labels=labels[j].detach().cpu().numpy()
                temp_preds=predictions[j].detach().cpu().numpy()
                print('temp_labels',temp_labels)
                print('temp_preds',temp_preds)
                list_paths.append(temp_paths)
                list_labels.append(temp_labels[0])
                list_preds.append(temp_preds[0])
                #list_p.append(p)


    dict = {'path':list_paths,'label':list_labels,'prediction':list_preds}
    df = pd.DataFrame(dict)
    df.to_csv(savepath+'predictions_AM_'+str(size)+'_'+str(myseed)+'_'+str(nb_classes)+'_'+str(data_type)+'.csv',index=False)
#####################################################################################
def classification_predictions(dataloader,dataset,size,model,savepath,seed,nb_classes=17,data_type='val'):
    was_training = model.training
    model.eval()
    list_paths=[]
    list_labels=[]
    list_preds=[]
  
    model=model.to('cpu')
    first_parameter = next(model.parameters())
    input_length = len(first_parameter.size())
    header_l=['path','true label','prediction']
    class_to_idx=dataset.class_to_idx
    print(class_to_idx)
    idx_to_class={v: k for k, v in class_to_idx.items()}
    with torch.no_grad():
        for i, (inputs,labels,paths) in enumerate(dataloader):
            if input_length> len(inputs.shape):
                inputs=inputs.unsqueeze(1)
            inputs=inputs.to('cpu')
            labels=labels.to('cpu')
            labels.numpy()
            predictions = model(inputs) #value of the output neurons
            _, pred= torch.max(predictions,1)
            for j in range(inputs.size()[0]):
                temp_paths=paths[j]
                temp_labels=labels[j].item()
                temp_preds=pred[j].numpy()
                temp_preds=int(temp_preds)
                temp_labels=int(temp_labels)
                real_pred=idx_to_class[temp_preds]
                real_label=idx_to_class[temp_labels]
                list_paths.append(temp_paths)
                list_labels.append(real_label)
                list_preds.append(real_pred)

    dict = {'path':list_paths,'label':list_labels,'prediction':list_preds}
    df = pd.DataFrame(dict)
    df.to_csv(savepath+'predictions_class_W_L'+str(size)+'_'+str(seed)+'_'+str(nb_classes)+'_'+str(data_type)+'_.csv',index=False)

    return

