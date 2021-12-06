#!/usr/bin/env python3

import timeit
import numpy as np

# My Classes
import libfileio as my_io

class RunKNN(object):
    """My Implementation of the KNN Algorithm"""

    def __init__(self,k=3):
        self.io = my_io.WrapperFileIO("../data/",None)
        self.k = k

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

    def cpu(self,x_train=None,y_train=None,x_eval=None,debug=True,run_count=0):
        """
        Runs the CPU version of the KNN classifier on the provided data. If no
        arguments are passed for x_train, y_train, or x_eval, the classifier
        will utilize the data already loaded by the class with the getData()
        routine.

        Setting the debug flag to [T] will cause statements to be printed to
        the command line. It is recommended to set this flag to [F] when timing

        If an argument is passed run_count, the classifier will terminate after
        classifing that many instances of the x_eval data set.
        """
        if debug: print("Running CPU Version")

        # Assign the Local Data (If applicable)
        if x_train is not None: self.x_train = x_train
        if y_train is not None: self.y_train = y_train
        if x_eval is not None: self.x_eval = x_eval

        # Get up Variables
        num_pts_train = self.x_train.shape[0]
        num_pts_eval = self.x_eval.shape[0]
        num_features = self.x_eval.shape[1]
        y_eval = np.zeros(num_pts_eval)

        # Visit each point to be evaluated
        for i in range(num_pts_eval):
            if debug:
                print("Training Point\n",self.x_train[i,:])
                print("Evaluation Point\n",self.x_eval[i,:])

            # Create an array to store the calculated distances
            distances = np.zeros(num_pts_train)
            for j in range(num_pts_train):
                # Calculate the distances
                sum = 0.0
                # Iterate over the feautures
                for n in range(num_features):
                    # Determine the delta between the eval and training features
                    delta = self.x_eval[i,n] - self.x_train[j,n]
                    # Increment the sum by the square of the features
                    sum += delta**2
                # Add the distance to the distance array
                distances[j] = sum**0.5

            # Sort the distances
            idx = np.argsort(distances)
            top_k = self.y_train[idx[0:self.k]]
            top_k = top_k[top_k>=0]
            if debug: print("TopK",top_k)

            # Vote and Assign Labels
            y_pred = np.bincount(top_k).argmax()
            y_eval[i] = y_pred
            if debug: print("Predicted:",y_pred,"Actual",self.y_train[i])

            # Early Termination, for testing only!
            if (run_count > 0) and (i > run_count): break


        # Return the labels
        self.y_eval = y_eval
        return y_eval


    def gpu(self):
        print("Running GPU Version")

    def lib(self):
        print("Running Library Version from Open3D")

if __name__ == '__main__':
    knn = RunKNN(k=5)
    knn.getData("pointcloud1.pickle","pointcloud2.pickle")
    knn.cpu(debug=True,run_count=15)
    try:
        print("NYI")

    except:
        print("ERROR, EXCEPTION THROWN")
