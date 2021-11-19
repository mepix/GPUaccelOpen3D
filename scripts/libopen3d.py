#!/usr/bin/env python3

import numpy as np
import open3d as o3d

class WrapperOpen3d(object):
    """This lightweight wrapper on Open3D converts a .ply point cloud into a NumPy Array"""

    def __init__(self, path_to_ply):
        self.path_to_ply = path_to_ply
        self.point_cloud_loaded = False
        return None

    def plyOpen(self,verbose=True):
        """Opens the point cloud from a .ply file specificied at class initialization"""
        if verbose: print("Opening",self.path_to_ply)
        self.pcd = o3d.io.read_point_cloud(self.path_to_ply)

        # Display Information About the Point Cloud
        if verbose: print(self.pcd)

        # Set the loaded flag to True
        self.point_cloud_loaded = True
        return None

    def visPointCloud(self):
        """Visualizes the loaded point cloud"""
        if self.point_cloud_loaded:
            o3d.visualization.draw_geometries([self.pcd])
        else:
            print("No Point Cloud Loaded, Cannot Visualize")
        return None

    def getNumPyPts(self):
        """Returns the opened point cloud as a NumPy Array"""
        if self.point_cloud_loaded:
            return np.asarray(self.pcd.points)
        else:
            print("No Point Cloud Loaded, Cannot Get Points")
            return None

    def getNumPyNorms(self):
        """Estimates the normals for reach point, returns as NumPy Array if True"""
        # self.pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        if self.point_cloud_loaded:
            return np.asarray(self.pcd.normals)
        else:
            print("No Point Cloud Loaded, Cannot Get Normals")
            return None

    def getNumPyColors(self):
        """Returns the opened point colors as a NumPy Array"""
        if self.point_cloud_loaded:
            return np.asarray(self.pcd.colors)
        else:
            print("No Point Cloud Loaded, Cannot Get Colors")
            return None

    def getNumpyAll(self,verbose=False):
        """Returns an array of [X,Y,Z,R,G,B,NormX,NormY,NormZ]"""
        # Get a Points
        npd = self.getNumPyPts()
        if verbose:
            print(npd.shape)
            print(npd)

        # Get Colors
        npc = self.getNumPyColors()
        if verbose:
            print(npc.shape)
            print(npc)

        # Get Normals
        npn = self.getNumPyNorms()
        if verbose:
            print(npn.shape)
            print(npn)

        # Concatenate Array
        npa = np.hstack((npd,npc,npn))
        if verbose:
            print(npa.shape)
            print(npa)

        # Output combined NumPy array
        return npa

if __name__ == '__main__':
    try:
        print("Testing Open3D")

        # Intialize the Class
        myOpen3d = WrapperOpen3d("../data/LeafPointCloud.ply")

        # Open the Point Cloud
        myOpen3d.plyOpen()

        # Visualize the Point Cloud
        myOpen3d.visPointCloud()

        # Get the NumPy Data (Pts,Colors,Normals)
        myNumPyArray = myOpen3d.getNumpyAll(verbose=True)

    except:
        print("ERROR, EXCEPTION THROWN")
