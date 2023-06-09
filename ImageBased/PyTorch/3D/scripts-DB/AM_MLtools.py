import torchvision
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
import numpy as np
import time
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
    size=int(((input.split('/')[-1]).split('-')[0]).split('Evec0')[1])
    functions=functions.reshape(size,size,size)
    functions=functions*functions
    return functions

#####################################################################################
def train_model(model,train,val,device,criterion, optimizer, num_epochs, scheduler,savepath, method,dataname,modelname,modelpath,batch_size,class_names):
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
                running_corrects+= torch.sum(preds==labels.data)
            if phase == train:
                scheduler.step()

            if phase == train:
                epoch_loss= running_loss/len(phase.dataset)
                epoch_acc = running_corrects.double()/ len(phase.dataset)
                print('{} loss= {:4f}, accuracy= {:4f}'.format(
                    'Training result:', epoch_loss, epoch_acc))
                accuracy.append(epoch_acc)
                _loss.append(epoch_loss)
                epochs.append(epoch)

            if phase == val:
                epoch_loss= running_loss/len(val.dataset)
                epoch_acc = running_corrects.double()/len(val.dataset)
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


