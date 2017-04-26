import pandas as pd
import numpy as np
import cv2
import os
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator

class LabelledImageProcessor:
    def __init__(self, directory, labels, target_size,
                 test_split=0.2, scale=True, verbose=True):
        X, self.names = self._read_from_dir(
            directory, target_size, verbose=verbose, scale=scale)
        y, self.labels = self._get_label_matrix(labels)
        
        # Store these for test data if we need to replicate preprocessing
        self.target_size = target_size
        self.scale = scale
        self.verbose = verbose
        
        self.X_train, self.X_test, self.y_train \
            ,self.y_test = train_test_split(X, y, test_size=test_split)

    def _read_from_dir(self, directory, target_size, scale, verbose):
        files = os.listdir(directory)

        X = np.zeros((len(files),) + target_size, dtype=np.float32)
        names = np.empty(len(files), dtype=object)

        for i, filename in enumerate(files):
            if i % (len(files) / 10) == 0 and i > 0 and verbose:
                print "{} images processed.".format(i)

            image = cv2.imread(directory + filename) 

            X[i, :] = cv2.resize(image, target_size[:2], 
                interpolation=cv2.INTER_AREA)
            names[i] = filename

        # We're going to use this generator as a preprocessor
        if scale:
            X, names = next(ImageDataGenerator(
                        rescale=1./255., 
                        samplewise_center=True
                        ).flow(
                        X, 
                        names, 
                        batch_size = len(X),
                        shuffle=False))
        
        return X, names

    def _get_label_matrix(self, labels):
        mlb = MultiLabelBinarizer()
        y = mlb.fit_transform([labels[name] for name in self.names])
        labels = mlb.classes_
        return y, labels

    def get_train_test_data(self, labels):
        idx = np.in1d(self.labels, labels)
        y_train = self.y_train[:, idx]
        y_test = self.y_test[:, idx]
        return self.X_train, self.X_test, y_train, y_test
        
    def get_test_data_from_dir(self, directory):
        return self.read_from_dir(directory, self.target_size, self.scale, self.verbose)
