#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 24 08:28:01 2018

@author: surbhikhandelwal
"""

# Data Preparation
import pandas as pd
import numpy as np

"""Load Data"""
train = pd.read_csv("train_DR.csv")
test = pd.read_csv("test_DR.csv")

Y_train = train["label"]

#Drop label coulmn
X_train = train.drop(labels =["label"], axis =1)

#train can be deleted now
del train

import matplotlib.pyplot as plt
import seaborn as sns

g = sns.countplot(Y_train)
Y_train.value_counts()

#Check for null and missing values
X_train.isnull().any().describe()
test.isnull().any().describe()

#no missing value found

""" NORMALIZATION- Perform grayscale normalization to reduce the effect of illumination's differences
Moreover, CNN converge faster on [0..1] data than [0..255]"""
X_train = X_train / 255.0
test = test  / 255.0
 
#Reshape image in 3 dimensions (height = 28px, width = 28px , length = 1) X*28*28*1 = 784

X_train = X_train.values.reshape(-1,28,28,1)
test = test.values.reshape(-1,28,28,1)

"""Label Encoding"""
# Encode labels to one hot vectors (ex : 2 -> [0,0,1,0,0,0,0,0,0,0])
from keras.utils.np_utils import to_categorical
Y_train = to_categorical(Y_train, num_classes = 10)


"""Split training and Validation Set"""
from sklearn.model_selection import train_test_split
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size =0.1, random_state= 0)

#Visualing the image
g = plt.imshow(X_train[0][:,:,0])

""" CNN """
#set the CNN model
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau
model = Sequential()

model.add(Conv2D(filters = 32, kernel_size = (5,5), padding = 'same', 
                 activation = 'relu', input_shape = (28,28,1)))
model.add(Conv2D(filters = 32, kernel_size = (5,5), padding = 'same', 
                 activation = 'relu', input_shape = (28,28,1)))
model.add(MaxPool2D(pool_size = (2,2)))
model.add(Dropout(0.25))



model.add(Conv2D(filters = 64, kernel_size = (3,3), padding = 'same', 
                 activation = 'relu', input_shape = (28,28,1)))
model.add(Conv2D(filters = 64, kernel_size = (3,3), padding = 'same', 
                 activation = 'relu', input_shape = (28,28,1)))
model.add(MaxPool2D(pool_size = (2,2), strides = (2,2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(256, activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation = 'softmax'))

""" Set the Optimizer & Analyzer"""
#define the optimizer
optimizer = RMSprop(lr = 0.001, rho = 0.9, epsilon=1e-08, decay = 0.0)

#compile the model
model.compile(optimizer = optimizer, loss= "categorical_crossentropy", metrics = ["accuracy"])

# setting the learning rate annealer to make the optimizer converge faster and closest to the global minimum
learning_rate_reduction = ReduceLROnPlateau(monitor= 'val_acc', patience = 3, verbose = 1,
                                            factor = 0.5, min_lr= 0.00001)

#Without Data Augmentation (accuracy = 98%)
history = model.fit(X_train, Y_train, batch_size = 86, epochs = 30, 
          validation_data = (X_val, Y_val), verbose = 2)



# With data augmentation to prevent overfitting (accuracy 0.99286)

datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.1, # Randomly zoom image 
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False)  # randomly flip images


datagen.fit(X_train)
batch_size=86

# Fit the model
history = model.fit_generator(datagen.flow(X_train,Y_train, batch_size=86),
                              epochs = 1, validation_data = (X_val,Y_val),
                              verbose = 2, steps_per_epoch=X_train.shape[0] // batch_size
                              , callbacks=[learning_rate_reduction])

# Predict the values from the validation dataset
Y_pred = model.predict(X_val)
# Convert predictions classes to one hot vectors 
Y_pred_classes = np.argmax(Y_pred,axis = 1) 
# Convert validation observations to one hot vectors
Y_true = np.argmax(Y_val,axis = 1) 

# predict results
results = model.predict(test)

# select the indix with the maximum probability
results = np.argmax(results,axis = 1)

results = pd.Series(results,name="Label")

submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)

submission.to_csv("cnn_datagen.csv",index=False)
































































