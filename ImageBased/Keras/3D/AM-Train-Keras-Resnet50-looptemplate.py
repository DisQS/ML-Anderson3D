#!/usr/bin/env python
# coding: utf-8

# # Anderson model of localization

print("--- parameter choices")

myseed= 1080
width= 100
nimages= 100

img_sizeX= 100; batch_size= 128
#img_sizeX= 500; batch_size= 12

img_sizeY= img_sizeX

num_epochs= 500
step_epoch= 2
validation_split= 0.1

mylr= 0.0001
mywd= 1e-6

#dataname='JPG-L'+str(width)+'-'+str(nimages)+'-s'+str(img_sizeX)
#dataname='Pet-L'+str(width)+'-'+str(nimages)+'-s'+str(img_sizeX)
dataname='L'+str(width)+'-'+str(nimages)+'-s'+str(img_sizeX)

#datapath = '/storage/disqs/'+'ML-Anderson3D/Images/'+dataname # SC-RTP
datapath = '/mnt/DataDrive/'+'ML-Anderson3D/Images/'+dataname # Ubuntu home RAR

print(dataname,"\n",datapath)

method='Keras-Resnet50-'+str(myseed)#+'-e'+str(num_epochs) #+'-bs'+str(batch_size)
print(method)

savepath = './'+dataname+'/'
import os
try:
    os.mkdir(savepath)
except FileExistsError:
    pass

print(savepath)
previousmodelpath='EMPTY'
previousmodelname='EMPTY'
save_train_loss= save_train_accuracy= save_valid_loss= save_valid_accuracy= []

print("--- initializations")

#standard notebook settings
#get_ipython().run_line_magic('reload_ext', 'autoreload')
#get_ipython().run_line_magic('autoreload', '2')
#get_ipython().run_line_magic('matplotlib', 'inline')

print("--- standard libraries")

import numpy as np
import pickle 
import sys
np.set_printoptions(threshold=sys.maxsize)

# PyCode tools
helperpath= '/storage/disqs/ML-Anderson3D/'+'ML-Anderson3D/PyCode' # SC-RTP
sys.path.insert(0,helperpath)

import random as rn
import os
import matplotlib.pyplot as plt
sys.path.insert(0,'../../../../PyCode/')

print("--- machine learning libraries")

import tensorflow as tf 
import keras
print("--- tensorflow: ",tf.__version__, ", keras: ", keras.__version__)

import sklearn
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
print("--- sklearn: ", sklearn.__version__)

print("--- special subroutines")

from keras.models import Sequential
from keras.layers import Dense, Conv2D
from keras.layers import Conv1D, MaxPooling2D
from keras.layers import AveragePooling2D, Flatten
from keras.layers import Dropout
from keras import optimizers
from keras.models import load_model

#from tensorflow.keras.models import load_model

print("--- starting the main code")

np.random.seed(myseed) # necessary for starting Numpy generated random numbers in a well-defined initial state.
rn.seed(myseed+1) # necessary for starting core Python generated random numbers in a well-defined state.

# The below is necessary in Python 3.2.3 onwards to
# have reproducible behavior for certain hash-based operations.
# See these references for further details:
# https://docs.python.org/3.4/using/cmdline.html#envvar-PYTHONHASHSEED
# https://github.com/fchollet/keras/issues/2280#issuecomment-306959926

os.environ['PYTHONHASHSEED'] = '0'

# Force TensorFlow to use single thread.
# Multiple threads are a potential source of
# non-reproducible results.
# For further details, see: https://stackoverflow.com/questions/42022950/which-seeds-have-to-be-set-where-to-realize-100-reproducibility-of-training-res

session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)

from tensorflow.keras import backend as K

# The below tf.set_random_seed() will make random number generation
# in the TensorFlow backend have a well-defined initial state.
# For further details, see: https://www.tensorflow.org/api_docs/python/tf/set_random_seed

tf.compat.v2.random.set_seed(myseed+3)
#tf.set_random_seed(1234)

#sess = tf.compat.v2.Session(graph=tf.get_default_graph(), config=session_conf)
#K.set_session(sess)

print("--- reading the images")

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1./255,validation_split=validation_split)
valid_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(datapath,
                                                 subset='training',
                                                 target_size = (img_sizeX,img_sizeY),
                                                 batch_size = batch_size, 
                                                 class_mode='categorical',
                                                 color_mode='rgba',
                                                 shuffle=True,seed=myseed)

validation_set= train_datagen.flow_from_directory(datapath, 
                                              subset='validation', 
                                              target_size = (img_sizeX,img_sizeY),
                                              batch_size = batch_size,
                                              class_mode='categorical',
                                              color_mode='rgba',
                                              shuffle=False,seed=myseed)

num_of_train_samples = training_set.samples
num_of_valid_samples = validation_set.samples
num_classes = len(validation_set.class_indices)

#print('--- Configure the dataset for performance')
#AUTOTUNE = tf.data.AUTOTUNE
#training_set = training_set.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
#validation_set = validation_set.cache().prefetch(buffer_size=AUTOTUNE)

fig=plt.figure(figsize=(10,5))
#for i in range(6):
for i in range(6):
#    plotsample=rn.sample(range(num_of_valid_samples),1)
#    plotsampleid=plotsample[0]
#    print(i,plotsampleid)
    plt.subplot(2,3,i+1)
    for x,y in validation_set:
        plt.imshow(x[0],cmap='hsv')
        #plt.title('y={}'.format(y[0]))
        #plt.title(labels[i])
        plt.axis('off')
        break
plt.tight_layout()
#plt.show()
plt.close()
fig.savefig(savepath+'/'+method+'_'+dataname+'_images'+'.png')

print("--- building the CNN")

from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
#from keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from keras.preprocessing import image

#resnet = ResNet50(include_top=False,weights='imagenet',input_shape=(img_sizeX, img_sizeY, 3))
resnet = ResNet50(include_top=False,weights=None,input_shape=(img_sizeX, img_sizeY, 4))

def create_CNN():
    # instantiate model
    model= Sequential([
        resnet,Flatten(),
        Dense(num_classes, activation='sigmoid'),
    ])
    
    return model

print('    CNN architecture (ResNet50) created successfully!')

print("--- Choosing the optimizer and the cost function")

#opt = optimizers.SGD(lr=mylr, decay=mywd)
opt = tf.keras.optimizers.Adam(learning_rate=mylr, decay=mywd)

def compile_model(optimizer=opt):
    # create the mode
    model=create_CNN()
    # compile the model
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=opt,
                  metrics=['accuracy','categorical_crossentropy'])
    return model

# opt = optimizers.SGD(lr=mylr, decay=mywd)
# model.compile(optimizer = opt, loss = 'categorical_crossentropy', metrics = ['accuracy'])

# create the deep neural net
model = compile_model()
print(model.summary())

print('--- model and parameter defined successfully, ready for training')

print("--- learning the images")

previousmodelloaded= False

for epochL in range(1,num_epochs,step_epoch):

    print('+++> epochs',epochL,'-',epochL+step_epoch-1,'of', num_epochs)

    #print(previousmodelpath,previousmodelloaded)

    modelname = method+'_model'+'_e'+str(epochL+step_epoch-1)+'_'+dataname+'.pth'
    historyname = method+'_history'+'_e'+str(epochL+step_epoch-1)+'_'+dataname+'.pkl'
    #print('--- training for',modelname,"\n",historyname)

    modelpath = savepath+modelname
    historypath = savepath+historyname

    print('--- initiating training for',modelname)
    
    if (os.path.isdir(modelpath) == True):
        print('---',modelname," exists, skipping ---!")
        previousmodelpath=modelpath
        previoushistorypath=historypath
        continue
    else:
        print('---',modelname,"does not yet exist --- needs training!")
        if(previousmodelloaded == True): # previous model is in memory already
            print('--- continuing with model as in memory')
            #continue
        elif(previousmodelpath != 'EMPTY'): #previous model not yet loaded in memory
            print('--- found', previousmodelpath, '--- loading!')
            model=load_model(previousmodelpath)
            previousmodelloaded= True
            
            # loading the history as well
            histfile=open(previoushistorypath,'rb')
            previous_history=pickle.load(histfile)
            save_train_loss= previous_history[0]
            save_train_accuracy= previous_history[1]
            save_valid_loss= previous_history[2]
            save_valid_accuracy= previous_history[3]
            histfile.close()

        else:
            print('--- previous model is ',previousmodelpath,", needs fresh restart!")

    # train DNN and store training info in history
    print('--- starting the training')
    history = model.fit_generator(training_set,
                                  steps_per_epoch = training_set.samples // batch_size,
                                  epochs = step_epoch,
                                  validation_data = validation_set,
                                  validation_steps = validation_set.samples // batch_size)

    # tf.keras.models.save_model(history,'Anderson_Ohtsuki_model_L20_500_keras_SGD_0_01_good_input_size.h5') 

    print('--- saving the current state to',modelpath)

    model.save(modelpath) 
    previousmodelpath=modelpath
    previousmodelname=modelname
    previousmodelloaded=True

    save_train_loss= save_train_loss + history.history['loss']
    save_valid_loss= save_valid_loss + history.history['val_loss']
    save_train_accuracy= save_train_accuracy + history.history['accuracy']
    save_valid_accuracy= save_valid_accuracy + history.history['val_accuracy']

    save_history=[save_train_loss,save_train_accuracy,save_valid_loss,save_valid_accuracy]

    histfile=open(historypath,"wb")
    pickle.dump(save_history,histfile)
    histfile.close()

    #history = load_model(modelpath)
    
    print("--- training history")

    #print(history.history)
    
    # evaluate model
    score=model.evaluate(validation_set,verbose=1)
    
    print("--- testing the quality of the learned model")
    
    # print performance
    print()
    print('--> loss:', score[0])
    print('--> accuracy:', score[1])
    
    # look into training history
    
    print("--- saving the plots of accuracy and loss")

    # summarize history for accuracy
    fig=plt.figure()
    #plt.plot(history.history['accuracy'])
    #plt.plot(history.history['val_accuracy'])
    plt.plot(save_train_accuracy)
    plt.plot(save_valid_accuracy)
    plt.ylabel('model accuracy')
    plt.xlabel('epoch')
    plt.legend(['train_acc', 'valid_acc'], loc='best')
    plt.title(dataname)
    #plt.show()
    plt.close()
    fig.savefig(savepath+'/'+method+'_e'+str(epochL+step_epoch-1)+'_'+dataname+'_accuracy'+'.png')
    
    # summarize history for loss
    fig=plt.figure()
    #plt.plot(history.history['loss'])
    #plt.plot(history.history['val_loss'])
    plt.plot(save_train_loss)
    plt.plot(save_valid_loss)
    plt.ylabel('model loss')
    plt.xlabel('epoch')
    plt.legend(['train_loss', 'valid_loss'], loc='best')
    plt.title(dataname)
    #plt.show()
    plt.close()
    fig.savefig(savepath+'/'+method+'_e'+str(epochL+step_epoch-1)+'_'+dataname+'_loss'+'.png')
    
    # summarize history for loss + accuracy
    fig=plt.figure()
    #plt.plot(history.history['loss'])
    #plt.plot(history.history['val_loss'])
    plt.plot(save_train_loss)
    plt.plot(save_valid_loss)
    plt.plot(save_train_accuracy)
    plt.plot(save_valid_accuracy)
    plt.ylabel('model accuracy+loss')
    plt.xlabel('epoch')
    plt.legend(['train_loss', 'valid_loss','train_acc', 'valid_acc'], loc='best')
    plt.title(dataname)
    #plt.show()
    plt.close()
    fig.savefig(savepath+'/'+method+'_e'+str(epochL+step_epoch-1)+'_'+dataname+'_accloss'+'.png')
    
    print("--> confusion matrix")
    
    validation_set.reset()
    label=validation_set.class_indices.keys()
    
    #Confusion Matrix 
    Y_pred = model.predict_generator(
        validation_set, num_of_valid_samples // batch_size+1, verbose=1)
    y_pred = np.argmax(Y_pred, axis=1)
    
    #basic confusion matrix
    cfm=confusion_matrix(validation_set.classes, y_pred)
    print(cfm)
    
    #print(os.getcwd())
    #os.chdir('../../../../PyCode/')
    #sys.path.insert(0,'../../../../PyCode/')
    from plot_confusion_matrix import *
    
    print(plot_confusion_matrix(confusion_matrix(validation_set.classes, y_pred),
                                label,savepath+'/'+method+'_e'+str(epochL+step_epoch-1)+
                                '_'+dataname+'_cfm'+'.png',
                                title='Confusion matrix for '+dataname,
                                cmap=None,
                                normalize=True))
    
