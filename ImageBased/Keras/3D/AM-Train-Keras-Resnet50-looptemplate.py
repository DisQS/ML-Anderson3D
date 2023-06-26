#!/usr/bin/env python
# coding: utf-8

# # Anderson model of localization

print("--- parameter choices")

myseed= 111111
width= 100
nimages= 100

#img_sizeX= 100; batch_size= 128
img_sizeX= 500; batch_size= 8

img_sizeY= img_sizeX

num_epochs= 6
step_epoch= 2
validation_split= 0.1

mylr= 0.01
mywd= 1e-6

dataname='L'+str(width)+'-'+str(nimages)+'-s'+str(img_sizeX)
datapath = '/storage/disqs/'+'ML-Anderson3D/Images/'+dataname # SC-RTP
#datapath = '/mnt/DataDrive/'+'ML-Data/Anderson/Images/'+dataname # Ubuntu home RAR
print(dataname,"\n",datapath)

method='Keras-Resnet50-'+str(myseed)+'-e'+str(num_epochs) #+'-bs'+str(batch_size)
print(method,"\n")

savepath = './'+dataname+'/'
import os
try:
    os.mkdir(savepath)
except FileExistsError:
    pass

print(savepath)

print("--- initializations")

#standard notebook settings
#get_ipython().run_line_magic('reload_ext', 'autoreload')
#get_ipython().run_line_magic('autoreload', '2')
#get_ipython().run_line_magic('matplotlib', 'inline')

print("--- standard libraries")

import numpy as np
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

fig=plt.figure(figsize=(10,5))
for i in range(6):
    plt.subplot(2,3,i+1)
    for x,y in validation_set:
        plt.imshow(x[0],cmap='hsv')
        #plt.title('y={}'.format(y[0]))
        plt.axis('off')
        break
plt.tight_layout()
#plt.show()
plt.close()
fig.savefig(savepath+'/'+method+'_'+dataname+'_images'+'.png')

print("--- building the CNN")

from keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from keras.preprocessing import image

#resnet = ResNet50(include_top=False,weights='imagenet',input_shape=(img_sizeX, img_sizeY, 4))
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
opt = keras.optimizers.Adam(lr=mylr, decay=mywd)

def compile_model(optimizer=opt):
    # create the mode
    model=create_CNN()
    # compile the model
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=optimizer,
                  metrics=['accuracy','categorical_crossentropy'])
    return model

# opt = optimizers.SGD(lr=mylr, decay=mywd)
# model.compile(optimizer = opt, loss = 'categorical_crossentropy', metrics = ['accuracy'])

# create the deep neural net
model = compile_model()
print(model.summary())

print('--- model compiled successfully and ready to be trained.')

print("--- learning the images")

for epochL in range(1,num_epochs,step_epoch):

    print('+++> epochL:',epochL, ' of ', num_epochs)

    # train DNN and store training info in history
    history = model.fit_generator(training_set,
                                  steps_per_epoch = training_set.samples // batch_size,
                                  epochs = step_epoch,
                                  validation_data = validation_set,
                                  validation_steps = validation_set.samples // batch_size)

    # tf.keras.models.save_model(history,'Anderson_Ohtsuki_model_L20_500_keras_SGD_0_01_good_input_size.h5') 

    print("--- saving the current state")

    modelname = 'Model_'+method+'_'+str(epochL+1)+'_'+dataname+'.pth'
    historyname = 'History_'+method+'_'+str(epochL+1)+'_'+dataname+'.pkl'
    print(method,"\n",modelname,"\n",historyname)

    modelpath = savepath+modelname
    historypath = savepath+historyname
    print(savepath,modelpath,historypath)

    model.save(modelpath) 
    
    import pickle 
    f=open(historypath,"wb")
    pickle.dump(history,f)
    f.close()
    
    print("--- testing the quality of the learned model")
    
    #history = load_model(modelpath)
    
    print("--- training history")
    
    # evaluate model
    score=model.evaluate(validation_set,verbose=1)
    
    # print performance
    print()
    print('--> loss:', score[0])
    print('--> accuracy:', score[1])
    
    # look into training history
    
    print("--- saving the plots of accuracy and loss")

    # summarize history for accuracy
    fig=plt.figure()
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.ylabel('model accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='best')
    plt.title(dataname)
    #plt.show()
    plt.close()
    fig.savefig(savepath+'/'+method+'_'+str(epochL+1)+'_'+dataname+'_accuracy'+'.png')
    
    # summarize history for loss
    fig=plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.ylabel('model loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='best')
    plt.title(dataname)
    #plt.show()
    plt.close()
    fig.savefig(savepath+'/'+method+'_'+str(epochL+1)+'_'+dataname+'_loss'+'.png')
    
    print("--> confusion matrix")
    
    validation_set.reset()
    label=validation_set.class_indices.keys()
    
    #Confusion Matrix 
    Y_pred = model.predict_generator(
        validation_set, num_of_valid_samples // batch_size+1, verbose=1)
    y_pred = np.argmax(Y_pred, axis=1)
    
    #basic confusion matrix
    confusion_matrix(validation_set.classes, y_pred)
    
    #print(os.getcwd())
    #os.chdir('../../../../PyCode/')
    #sys.path.insert(0,'../../../../PyCode/')
    from plot_confusion_matrix import *
    
    print(plot_confusion_matrix(confusion_matrix(validation_set.classes, y_pred),
                                label,savepath+'/'+method+'_'+str(epochL+1)+
                                '_'+dataname+'_cfm'+'.png',
                                title='Confusion matrix for '+dataname,
                                cmap=None,
                                normalize=True))
    
