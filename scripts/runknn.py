#!/usr/bin/env python3

import numpy as np
from numba import cuda, float32, uint16, int32 # GPU Optimizations
import math
import copy

# My Classes
import libfileio as my_io
import libtimer as my_timer

@cuda.jit
def increment_a_2D_array(an_array):
    x, y = cuda.grid(2)
    if x < an_array.shape[0] and y < an_array.shape[1]:
       an_array[x, y] += 1

# https://stackoverflow.com/questions/41769100/how-do-i-use-numba-on-a-member-function-of-a-class
TPB = 256 # Threads Per Block
MAX_POINTS = 1280
NUM_FEATURES = 9
K_NEAREST = 5

# Numba Datatypes: https://numba.pydata.org/numba-doc/latest/reference/types.html

@cuda.jit
def kernelKNN(x_train,y_train,x_eval,y_eval):
    """
    This is a rather naive implementation of a KNN Alogrithm:
    -> Each thread should correspond to a point in the x_eval point cloud
    -> Each thread calculates the distance between it's x_eval and all points
       in x_train
    """

    # Get the Thread ID and Stride
    tx = cuda.threadIdx.x
    stride = cuda.blockIdx.x *cuda.blockDim.x

    # Check Boundaries
    if stride + tx > x_eval.shape[0]:
        return

    # Load the training point cloud into shared memory
    x_train_shared = cuda.shared.array(shape=(MAX_POINTS,NUM_FEATURES), dtype=float32)

    for j in range(NUM_FEATURES):
        x_train_shared[stride+tx,j] = x_train[stride+tx,j]
    # Wait until all threads finish preloading
    cuda.syncthreads()

    # Create an array to store the distances for this block
    distances = cuda.local.array(shape=(MAX_POINTS,1), dtype=float32)
    # Each thread corresponds to a point in the evaluation array
    # Calculate the distances for this point w.r.t all the points in the train
    for j in range(x_train.shape[0]):
        # Calculate the distances
        sum = 0.0
        # Iterate over the feautures
        for n in range(NUM_FEATURES):
            # Determine the delta between the eval and training features
            delta = x_eval[stride+tx,n] - x_train_shared[j,n]
            # Increment the sum by the square of the features
            sum += delta**2

        distances[j,0] = sum**0.5

    # Wait until all threads finish preloading
    cuda.syncthreads()

    # FOR DEBUG OF DISTANCES ONLY
    # y_eval[stride+tx] = distances[0,0]
    # return

    # Make a local copy of the labels so not all threads are trying to sort at once!
    y_train_copy = cuda.local.array(shape=(MAX_POINTS,1), dtype=int32)
    for i in range(MAX_POINTS):
        if i < y_train.shape[0]:
            y_train_copy[i,0] = y_train[i]
        else:
            y_train_copy[i,0] = 0

    # Sort the distances (Selection Sort)
    # TODO: consider breaking this out into another kernel
    # SEE: https://stackoverflow.com/questions/43962195/selection-sort-in-cuda
    # It seems like either a radix or a bucket sorts is better fit for GPU
    # SEE: https://en.wikipedia.org/wiki/Radix_sort
    # SEE: https://stackoverflow.com/questions/34023841/vectorized-radix-sort-with-numpy-can-it-beat-np-sort
    for i in range(distances.shape[0]-1):
        min_val = distances[i,0]
        min_idx = i
        y_val = y_train_copy[i,0]
        for j in range(i+1,distances.shape[0]):
            if distances[j,0] < min_val:
                min_val = distances[j,0]
                min_idx = j
                y_val = y_train_copy[j,0]
        distances[min_idx,0] = distances[i,0]
        distances[i,0] = min_val

        y_train_copy[min_idx] = y_train_copy[i,0]
        y_train_copy[i,0] = y_val

    # Select the top K labels
    offset = MAX_POINTS - y_train.shape[0] #Offset need since there are padded zeros
    # print(offset)
    # print(y_train_copy[offset,0])
    top_k = y_train_copy[offset:offset+K_NEAREST,0]
    # print(top_k[0],top_k[1],top_k[2],top_k[3],top_k[4])

    # Vote and Assign Labels
    counter = cuda.local.array(shape=(K_NEAREST,1), dtype=uint16)
    count_max = 0
    y_pred = 0
    for i in range(top_k.shape[0]):
        val_current = top_k[i]
        count_now = 0
        for j in range(top_k.shape[0]):
            if (val_current == top_k[j]):
                count_now += 1
                if (count_now > count_max):
                    count_max = count_now
                    y_pred = top_k[j]
        counter[i] = count_now

    # load back into y_eval
    y_eval[stride+tx] = y_pred#distances[0,0]#y_pred #TODO: fix stride

class RunKNN(object):
    """My Implementation of the KNN Algorithm"""

    def __init__(self,k=3,path_to_data="../data/",threads_per_block=256):
        self.io = my_io.WrapperFileIO(path_to_data,None)
        self.k = k
        self.tpb = threads_per_block
        self.x_train = None
        self.y_train = None
        self.x_eval = None
        self.y_eval = None

    def getData(self,file_pickle_train, file_pick_eval,verbose=False):
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
            if top_k.size ==0: # check if empty
                y_pred = -1
            else:
                y_pred = np.bincount(top_k).argmax()
            y_eval[i] = y_pred
            if debug: print("Predicted:",y_pred,"Actual",self.y_train[i])

            # Early Termination, for testing only!
            if (run_count > 0) and (i > run_count): break

        # Return the labels
        self.y_eval = y_eval
        return y_eval

    def gpu(self,x_train=None,y_train=None,x_eval=None,debug=False):
        print("Running GPU Version")

        # Assign the Local Data (If applicable)
        if x_train is not None: self.x_train = x_train
        if y_train is not None: self.y_train = y_train
        if x_eval is not None: self.x_eval = x_eval
        print("X_train Shape",self.x_train.shape)
        print("X_eval Shape",self.x_eval.shape)

        # Explicitly Create Numbas Types for the DEVICE
        self.y_eval = np.zeros(self.x_eval.shape[0])
        y_train_copy = copy.deepcopy(self.y_train)

        d_x_train = cuda.to_device(self.x_train)#,dtype=float32)
        d_y_train = cuda.to_device(self.y_train)#,dtype=int32)
        d_x_eval = cuda.to_device(self.x_eval)#,dtype=float32)
        d_y_eval = cuda.to_device(self.y_eval)#,dtype=int32)

        # Set up the Kernel
        threadsperblock = (self.tpb, 1)
        print("Threads Per Block",threadsperblock)
        blockspergrid_x = math.ceil(self.x_train.shape[0] / threadsperblock[0])
        blockspergrid_y = 1
        blockspergrid = (blockspergrid_x, blockspergrid_y)
        print("Blocks Per Grid",blockspergrid)

        # Launch the Kernel
        print("Launching GPU Kernel")
        kernelKNN[blockspergrid,threadsperblock](d_x_train,d_y_train,d_x_eval,d_y_eval)

        # Copy back to the host
        self.y_eval = d_y_eval.copy_to_host()

        if debug:
            # Write Labels to CSV File for Analysis
            self.io.saveCSV(self.y_eval,"test.csv")

            print(y_train_copy)
            print(self.y_train)
            print(d_y_train.copy_to_host())

    def gpu_test(self):
        print("Running GPU Test")
        try:
            threadsperblock = (16, 16)
            an_array = np.array([[1,2,3],[4,5,6]])
            print(an_array)
            blockspergrid_x = math.ceil(an_array.shape[0] / threadsperblock[0])
            blockspergrid_y = math.ceil(an_array.shape[1] / threadsperblock[1])
            blockspergrid = (blockspergrid_x, blockspergrid_y)
            print(blockspergrid)
            increment_a_2D_array[blockspergrid, threadsperblock](an_array)
            print(an_array)
        except:
            print("GPU ERROR")

    def lib(self):
        print("Running Library Version from Open3D")

if __name__ == '__main__':
    # Intialize the KNN Classifier
    knn = RunKNN(k=K_NEAREST,threads_per_block=TPB)

    # Initialize the Timer
    code_timer = my_timer.MyTimer()
    code_timer.start()

    # Open the Point Cloud files for Training and Evaluation
    knn.getData("pointcloud1.pickle","pointcloud2.pickle")
    print("Pickle Load Time:",code_timer.lap())

    # Run the CPU Implementation
    knn.cpu(debug=False,run_count=0)
    print("CPU Run Time:",code_timer.lap())
    knn.saveData("pointcloud-cpu.pickle")
    print("CPU Pickle Save Time",code_timer.lap())

    # Run the GPU Implementation
    knn.gpu()
    print("GPU Run Time:",code_timer.lap())
    knn.saveData("pointcloud-gpu.pickle")
    print("GPU Pickle Save Time",code_timer.lap())

    print("Total Run Time:",code_timer.ellapsed())
