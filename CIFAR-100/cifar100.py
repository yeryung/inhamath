# code from https://towardsdatascience.com/cifar-100-transfer-learning-using-efficientnet-ed3ed7b89af2
# by Chetna Khanna

import pickle
import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
# from pylab import rcParams
# import tensorflow as tf
# import keras
# %matplotlib inline
# from keras.models import Sequential, load_model
# from keras.layers import Conv2D, MaxPool2D, Dropout, Flatten, Dense, GlobalAveragePooling2D
# from keras.layers.normalization import BatchNormalization
# from keras.preprocessing.image import ImageDataGenerator
# from keras.optimizers import Adam, SGD
# from keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau
# from keras.utils import to_categorical
# from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
# from skimage.transform import resize
# from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
# import seaborn as sns
# import cv2
# import albumentations as albu

# get the data
def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='latin1')
    return dict

relPath = "cifar-100-python/"
trainData = unpickle(relPath+"train")
testData = unpickle(relPath+"test")
metaData = unpickle(relPath+"meta")

#storing coarse labels along with its number code in a dataframe
category = pd.DataFrame(metaData['coarse_label_names'], columns=['SuperClass'])
#storing fine labels along with its number code in a dataframe
subCategory = pd.DataFrame(metaData['fine_label_names'], columns=['SubClass'])

X_train = trainData['data']
print(type(X_train))

#4D array input for building the CNN model using Keras
X_train = X_train.reshape(len(X_train), 3, 32, 32).transpose(0,2,3,1)