import pandas as pd
import numpy as np
import cv2
import os
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import TrainTestSplit

class LabelledImageProcessor:
    def __init__(self, directory, labels, target_size=(128,128,3),
                 test_split=0.2, verbose=True):
        X, self.names = self._read_from_dir(directory, target_size, verbose)
        y, self.labels = self._get_label_matrix(labels)
        
        self.X_train, self.X_test, self.y_train, 
            self.y_test = train_test_split(X, y, test_size=test_split)

    def _read_from_dir(self, directory, target_size, verbose):
        files = os.listdir(directory)

        X = np.zeros((len(files),) + target_size, dtype=np.float32)
        names = np.empty(len(files), dtype=object)

        for i, filename in enumerate(files):
            if i % (len(files) / 10) == 0 and i > 0 and verbose:
                print "{} images processed.".format(i)

            image = cv2.imread(directory + image) 

            X_test_data[i, :] = cv2.resize(image, target_size[:2], 
                interpolation=cv2.INTER_AREA)
            names[i] = image
        
        return X, names

    def _get_label_matrix(self, labels):
        mlb = MultiLabelBinarizer()
        y = mlb.fit_transform([labels[name] for name in self.names])
        labels = mlb.classes_
        return y, labels

    def get_train_test_data(self, classes):
        idx = np.in1d(self.classes, classes)
        y_train = self.y_train[idx]
        y_test = self.y_test[idx]
        return self.X_train, self.X_test, y_train, y_test
        
