#!/usr/bin/env python3

# Python Imports
import numpy as np

# My Classes
import libopen3d as my_pc
import libfileio as my_io

def picklePointCloud(path_to_data,file_name_ply,file_name_pickle,do_get_labels,visualize=False,verbose=True):
    # Open the Point Cloud at the Specified file path
    pctool = my_pc.WrapperOpen3d(path_to_data +file_name_ply)
    pctool.plyOpen()
    if visualize: pctool.visPointCloud()

    # Convert it to a NumPy Array
    data_numpy = pctool.getNumpyAll()
    if verbose: print(data_numpy.shape)

    # Get the Labels
    if do_get_labels:
        data_labels = pctool.getClusterLabels()
        if verbose: print(data_labels.shape)
    else:
        data_labels = None

    # Build a dictionary from the points and labels
    data_dict ={
        "points" : data_numpy,
        "labels" : data_labels
    }
    if verbose: print(data_dict)

    # Save as a pickle file
    iotool = my_io.WrapperFileIO(path_to_data,file_name_pickle)
    iotool.savePickle(data_dict)

    return None

def visualizePointCloudPickle(path_to_data,file_name_pickle,color_clusters=False,verbose=True):
    # Open the Pickle File
    iotool = my_io.WrapperFileIO(path_to_data,file_name_pickle)
    data_dict = iotool.loadPickle()

    # Split the Dictionary
    data_numpy = data_dict['points']
    data_labels = data_dict['labels']

    # Convert to a Point Cloud Object
    pctool = my_pc.WrapperOpen3d(None)
    pcd = pctool.convertNumPyToPointCloud(data_numpy)


    if color_clusters:
        pctool.visPointCloudClusters(pcd,data_labels)
    else:
        pctool.loadPointCloud(pcd)
        pctool.visPointCloud()

    return None

if __name__ == '__main__':
    # Present the User with a set of run time option
    option = ""
    while option not in ['1','2','3','4','q','Q']:
        print("============ Select The Desired Point Cloud Operation ============")
        print("1) Transform the point cloud (.PLY) to a pickle file")
        print("2) Load and display the pickle files as point clouds")
        print("3) Load and display the pickle files with colored clusters")
        print("press \'q\' to quit")
        option = input("Enter Desired Option: ")
        print("Selected Option:",option)

    # Quit the Program
    if option in ['q','Q']:
        exit()

    # Option 1: Convert the point clouds to pickle files
    if option in ['1']:
        try:
            picklePointCloud("../data/","LeafPointCloud.ply","pointcloud1.pickle",True)
            picklePointCloud("../data/","LeafPointCloud.ply","pointcloud2.pickle",False)
        except:
            print("ERROR, EXCEPTION THROWN")
        exit()

    # Option 2: Convert the pickle files to point cloud objects
    if option in ['2']:
        visualizePointCloudPickle("../data/","pointcloud1.pickle")

    # Option 3: Convert the pickle files to point cloud objects with COLOR!
    if option in ['3']:
        visualizePointCloudPickle("../data/","pointcloud1.pickle",color_clusters=True)
        visualizePointCloudPickle("../data/","pointcloud-cpu.pickle",color_clusters=True)

    # Future Options
    if option in ['4']:
        print("***::TODO::NYI::***")
        exit()
