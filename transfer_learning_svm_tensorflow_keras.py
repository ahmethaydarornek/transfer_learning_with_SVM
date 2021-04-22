# -*- coding: utf-8 -*-
"""
Created on Thu Apr 22 09:26:38 2021

@author: ahmethaydarornek

A CNN model has two parts; first part is convolutional layer which extract
features from images and second part is neural layer which classifies the
extracted features.

It is known that an SVM model classifies images with more accuracy than 
neural layer.

In this script, we use a pre-trained CNN model as convolutional layer
and an SVM model for classification.
"""

import tensorflow # for pre-trained model 
import sklearn # for SVM classifier
import pickle # for saving the SVM model
import numpy # for matrix operations

# VGG16 is a pre-trained CNN model. 
conv_base = tensorflow.keras.applications.VGG16(weights='imagenet',
                  include_top=False,
                  input_shape=(224, 224, 3)
                  )

# Showing the convolutional layers.
conv_base.summary()

# Defining the directories that data are in.
train_dir = 'data/train'
validation_dir = 'data/val'

datagen = tensorflow.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
batch_size = 20

def extract_features(directory, sample_count):
    features = numpy.zeros(shape=(sample_count, 4, 4, 512))
    labels = numpy.zeros(shape=(sample_count))
    generator = datagen.flow_from_directory(
        directory,
        target_size=(150, 150),
        batch_size=batch_size,
        class_mode='binary')
    i = 0
    for inputs_batch, labels_batch in generator:
        features_batch = conv_base.predict(inputs_batch)
        features[i * batch_size : (i + 1) * batch_size] = features_batch
        labels[i * batch_size : (i + 1) * batch_size] = labels_batch
        i += 1
        if i * batch_size >= sample_count:
            # Note that since generators yield data indefinitely in a loop,
            # we must `break` after every image has been seen once.
            break
    return features, labels

train_features, train_labels = extract_features(train_dir, 300)
train_features = numpy.reshape(train_features, (300, 4 * 4 * 512))

# An SVM model (support vector classifier) is created.
clf = sklearn.svm.SVC()
# The created SVM classifer is trained.
clf.fit(train_features, train_labels)
# The trained SVM classifier is saved into the working directory.
filename = 'SVM_classifier.sav'
pickle.dump(clf, open(filename, 'wb'))
