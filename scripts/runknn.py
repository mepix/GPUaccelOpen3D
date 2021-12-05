#!/usr/bin/env python3

import timeit
import numpy as np

# My Classes
import libfileio as my_io

class RunKNN(object):
    """My Implementation of the KNN Algorithm"""

    def __init__(self,k=3):
        self.io = my_io.WrapperFileIO("../data/",None)

    def getData(self,file_pickle_train, file_pick_eval,verbose=True):
        """
        Opens the pickle files corresponing to the training and evaluation point
        cloud.
        """
        # Open the Pickle Files
        train_dict = self.io.loadPickle(file_pickle_train)
        eval_dict = self.io.loadPickle(file_pick_eval)
        if verbose:
            print(train_dict)
            print(eval_dict)

        # Split the Dictionaries
        x_train = train_dict["points"]
        y_train = train_dict["labels"]
        x_eval = eval_dict["points"]

        # Perform any Preprocessing / Normalization
        x_train = self.procData(x_train)
        x_eval = self.procData(x_eval)
        if verbose:
            print(x_train)
            print(x_eval)

        # Assign to Class Variables
        self.x_train = x_train
        self.y_train = y_train
        self.x_eval = x_eval

        return x_train, y_train, x_eval

    def procData(self,data):
        """
        Preprocesses the data before the KNN

        Data is in the form [X,Y,Z,R,G,B,NormX,NormY,NormZ]
        Data array content is defined in libopen3D

        Normalization should occur for each column to aid the classification
        """
        # Normalize the numpy array by column
        data_norm = (data-data.min(0)) / data.ptp(0)

        return data_norm

    def cpu(self):
        print("Running CPU Version")

    def gpu(self):
        print("Running GPU Version")

    def lib(self):
        print("Running Library Version from Open3D")

if __name__ == '__main__':
    knn = RunKNN()
    knn.getData("pointcloud1.pickle","pointcloud2.pickle")
    try:
        print("NYI")

    except:
        print("ERROR, EXCEPTION THROWN")
