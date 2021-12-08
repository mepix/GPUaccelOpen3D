#!/usr/bin/env python3

import numpy as np
from numba import cuda, float32, uint16, int32 # GPU Optimizations
import math
import copy

# My Classes
import libfileio as my_io
import libtimer as my_timer

NUM_ITERATIONS = 100
DISTANCE_THRESHOLD = 1
THREADS_PER_BLOCK = 256

class RunRANSAC(object):
    """docstring forRunRANSAC."""

    def __init__(self,num_iterations,distance_thresh,path_to_data="../data/",threads_per_block=256):
        # super(RunRANSAC, self).__init__()
        self.io = my_io.WrapperFileIO(path_to_data,None)
        self.num_iters = num_iterations
        self.dist_thresh = distance_thresh
        self.tpb = threads_per_block
        self.x_train = None
        self.y_train = None
        self.x_eval = None
        self.y_eval = None

    def getData(self,file_pickle_train, verbose=False):
        """
        Opens the pickle files corresponing to the point cloud.
        """
        # Open the Pickle Files
        train_dict = self.io.loadPickle(file_pickle_train)
        if verbose:
            print(train_dict)

        # Split the Dictionaries
        x_train = train_dict["points"]
        y_train = train_dict["labels"]

        # Perform any Preprocessing / Normalization
        x_train = self.procData(x_train)
        if verbose:
            print(x_train)

        # Assign to Class Variables
        self.x_train = x_train
        self.y_train = y_train

        return x_train, y_train

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

    def saveData(self,file_name,verbose=False):
        """Saves the processes data as a pickle file"""
        # Process the data as a dictionary
        data_dict = {
            "points" : self.x_eval,
            "labels" : self.y_eval
        }
        if verbose: print(data_dict)

        # Save the File
        self.io.savePickle(data_dict,file_name)
        return None

    def selectCluster(self,point_cloud_array,label_array,verbose=True):
        """
        This function selects the cluster corresponing to the most frequently
        occuring label in the point cloud
        """

        # Remove the negative labels
        if verbose: print(label_array.shape,label_array)
        label_clean = label_array[label_array>=0]
        if verbose: print(label_clean.shape,label_clean)

        # Determine the Biggest Cluster
        max_label = np.bincount(label_clean).argmax()
        if verbose: print("Most Frequent Label:",max_label)

        # Store Internally as the evaluation data
        self.x_eval = point_cloud_array[label_array==max_label]
        self.y_eval = np.full(self.y_train.shape,max_label)

        # Return the max label
        return max_label

    def cpu(self,x_train=None,y_train=None,x_eval=None,debug=True,run_count=0):
        """
        Runs the CPU version of the RANSAC algorithm on the provided data. If no
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

        # Get up Variables
        # num_pts_train = self.x_train.shape[0]
        # y_eval = np.zeros(num_pts_eval)

        # Perform the Iterations
        points_on_plane = []
        for i in range(self.num_iters):
            candidate_points = np.zeros(3)



if __name__ == '__main__':
    # Intialize the KNN Classifier
    ransac = RunRANSAC(
        num_iterations = NUM_ITERATIONS,
        distance_thresh = DISTANCE_THRESHOLD,
        threads_per_block=THREADS_PER_BLOCK)

    # Initialize the Timer
    code_timer = my_timer.MyTimer()
    code_timer.start()

    # Open the Point Cloud files for Training and Evaluation
    ransac.getData("pointcloud1.pickle")
    print("Pickle Load Time:",code_timer.lap())

    # Select a leaf
    ransac.selectCluster(ransac.x_train,ransac.y_train)
    ransac.saveData("pointcloud-ransac.pickle")
    print("CPU Pickle Save Time",code_timer.lap())


    # Run the CPU Implementation
    ransac.cpu(debug=False,run_count=0)
    print("CPU Run Time:",code_timer.lap())

    # Run the GPU Implementation
    # knn.gpu()
    # print("GPU Run Time:",code_timer.lap())
    # knn.saveData("pointcloud-gpu.pickle")
    # print("GPU Pickle Save Time",code_timer.lap())

    print("Total Run Time:",code_timer.ellapsed())

    try:
        print("NYI")

    except:
        print("ERROR, EXCEPTION THROWN")
