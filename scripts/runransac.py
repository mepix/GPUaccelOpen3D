#!/usr/bin/env python3

import numpy as np
from numba import cuda, float32, uint16, int32 # GPU Optimizations
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_uniform_float32
import math
import copy

# My Classes
import libfileio as my_io
import libtimer as my_timer

NUM_ITERATIONS = 100
DISTANCE_THRESHOLD = 1#0.1
THREADS_PER_BLOCK = 256

"""
Kernel 1: Calculate Constants a,b,c,d from the point cloud
    IN: [full_point_cloud]
    OUT: [a,b,c,d]
Kernel 2: Evaluate the fit for points a,b,c,d
    IN: [point cloud], [a,b,c,d]
    OUT: [inlier_points]
"""

@cuda.jit
def kernelRANSAC_1(point_cloud,plane_constants):
    """
    Calculates the constants a,b,c,d from the point_cloud that fit the plane
    equation: ax + by + cz + d = 0
    """

    # Get the current thread ID
    tx = cuda.threadIdx.x
    stride = cuda.blockIdx.x*cuda.blockDim.x

    # Check Bounds
    if stride + tx > point_cloud.shape[0]:
        return

    # Create an Array to store the point index
    rand_idx = cuda.local.array(shape=(3,1),dtype=int32)
    pts = cuda.local.array(shape=(3,3),dtype=float32)
    for i in range(3):
        rand_idx  = -1 #TODO: GET RANDOM NUMBER ON GPU
        # Get the Points Corresponding to the random indexs
        pts[i,0] = point_cloud[idx,0]
        pts[i,1] = point_cloud[idx,1]
        pts[i,3] = point_cloud[idx,2]

    # Calculate the Constants for the given points

    # $$ a = [(y_2 - y_1)(z_3 - z_1) - (z_2 - z_1)(y_3 - y_2)] $$
    a = (pts[1,1]-pts[0,1])*(pts[2,2]-pts[0,2]) - (pts[1,2]-pts[0,2])*(pts[2,1]-pts[1,1])

    # $$ b = [(z_2 - z_1)(x_3 - x_1) - (x_2 - x_1)(z_3 - z_2)] $$
    b = (pts[1,2]-pts[0,2])*(pts[2,0]-pts[0,0]) - (pts[1,0]-pts[0,0])*(pts[2,2]-pts[1,2])

    # $$ c = [(x_2 - x_1)(y_3 - y_1) - (y_2 - y_1)(x_3 - x_2)] $$
    c = (pts[1,0]-pts[0,0])*(pts[2,1]-pts[0,1]) - (pts[1,1]-pts[0,1])*(pts[2,0]-pts[1,0])

    # $$ d = -(a * x_n + b * y_n + c * z_n) $$
    d = -(a*pts[0,0] + b*pts[0,1] + c*pts[0,2])

    # $$ psq = sqrt{(a^2) + (b^2 + (c^2)} $$
    psq = max(0.1,sqrt(a*a + b*b + c*c))

    # Pass the Points back to the output array
    plane_constants[stride+tx,0] = a
    plane_constants[stride+tx,1] = b
    plane_constants[stride+tx,2] = c
    plane_constants[stride+tx,3] = d
    plane_constants[stride+tx,4] = psq  

    return None

@cuda.jit
def kernelRANSAC(x_train,y_train,x_eval,y_eval):
    return None

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

    def cpu(self,x_eval=None,y_eval=None,debug=True):
        """
        Runs the CPU version of the RANSAC algorithm on the provided data. If no
        arguments are passed for x_eval or y_eval, the algorithm will utilize
        the data already loaded by the class with the getData() routine.

        Setting the debug flag to [T] will cause statements to be printed to
        the command line. It is recommended to set this flag to [F] when timing

        THE MATH

        A plane is defined as:

        $$ ax + by + cz + d = 0 $$

        Given three points, the constants can be determined:

        $$ a = [(y_2 - y_1)(z_3 - z_1) - (z_2 - z_1)(y_3 - y_2)] $$
        $$ b = [(z_2 - z_1)(x_3 - x_1) - (x_2 - x_1)(z_3 - z_2)] $$
        $$ c = [(x_2 - x_1)(y_3 - y_1) - (y_2 - y_1)(x_3 - x_2)] $$
        $$ d = -(a * x_n + b * y_n + c * z_n) $$

        Consider all the other points in the point cloud and calculate the
        distance to the fit plane

        $$distance = \frac{a*x_4 + b*y_4 + c*z_5 + d}{\sqrt{a^2 + b^2 + c^2}}$$

        REFERENCES
        https://en.wikipedia.org/wiki/Random_sample_consensus
        https://medium.com/@ajithraj_gangadharan/3d-ransac-algorithm-for-lidar-pcd-segmentation-315d2a51351
        """
        if debug: print("Running CPU Version")

        # Assign the Local Data (If applicable)
        if x_eval is not None: self.x_eval = x_eval
        if y_eval is not None: self.y_eval = y_eval

        # Get up Variables
        # num_pts_train = self.x_train.shape[0]
        # y_eval = np.zeros(num_pts_eval)

        # Perform the Iterations
        pts_idx_inliers = [] # For points already on the plane
        pts_best = []
        constants_best = []
        for i in range(self.num_iters):
            # Get a Random Sampling of Candidate Indexs
            pts_idx = np.random.randint(0,self.x_eval.shape[0],(3))
            # pts_idx_inliers(pts_idx)

            if debug: print("Candidate Points Index:\n",pts_idx)

            # Get the Candidate Points
            pts = self.x_eval[pts_idx,:]
            if debug: print("Candidate Points:\n",pts)

            # Calculate the Constants for the given points

            # $$ a = [(y_2 - y_1)(z_3 - z_1) - (z_2 - z_1)(y_3 - y_2)] $$
            a = (pts[1,1]-pts[0,1])*(pts[2,2]-pts[0,2]) - (pts[1,2]-pts[0,2])*(pts[2,1]-pts[1,1])

            # $$ b = [(z_2 - z_1)(x_3 - x_1) - (x_2 - x_1)(z_3 - z_2)] $$
            b = (pts[1,2]-pts[0,2])*(pts[2,0]-pts[0,0]) - (pts[1,0]-pts[0,0])*(pts[2,2]-pts[1,2])

            # $$ c = [(x_2 - x_1)(y_3 - y_1) - (y_2 - y_1)(x_3 - x_2)] $$
            c = (pts[1,0]-pts[0,0])*(pts[2,1]-pts[0,1]) - (pts[1,1]-pts[0,1])*(pts[2,0]-pts[1,0])

            # $$ d = -(a * x_n + b * y_n + c * z_n) $$
            d = -(a*pts[0,0] + b*pts[0,1] + c*pts[0,2])

            # $$ psq = sqrt{(a^2) + (b^2 + (c^2)} $$
            psq = max(0.1,np.sqrt(a*a + b*b + c*c))

            if debug: print(a,b,c,d)

            # Evaluate the Performance of the Fit
            for j in range(self.x_eval.shape[0]):
                # We do not want to compare with points used to calc a,b,c,d
                # if j in pts_idx: continue

                # Calc distance between point and plane
                if debug: print(j)
                dist = math.fabs(a*pts[j,0] + b*pts[j,1] + c*pts[j,2] + d)/psq

                # Check whether or no the point is a good fit
                if (dist <= self.dist_thresh):
                    # Add to the list if inlier points
                    pts = np.vstack((pts,self.x_eval[j,:])) #TODO:DEBUG: THIS SHOULD BE A SEPARATE POINTS ARRAY

            # Check if the current inliers is better than the best so far
            if len(pts) > len(pts_best):
                pts_best = pts
                constants_best = np.array([a,b,c,d])

        return pts_best, constants_best


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
    _, plane_constants = ransac.cpu(debug=False)
    print("CPU Run Time:",code_timer.lap())
    print("CPU Plane Constants:",plane_constants)
    code_timer.lap()

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
