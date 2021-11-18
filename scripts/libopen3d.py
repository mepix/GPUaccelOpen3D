#!/usr/bin/env python3

import numpy as np
import open3d as o3d

class WrapperOpen3d(object):
    """docstring for ."""

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

    def getNumPyData(self):
        """Returns the opened point cloud as a NumPy Array"""
        if self.point_cloud_loaded:
            return np.asarray(self.pcd.points)
        else:
            print("No Point Cloud Loaded, Cannot Get NumPy Array")
            return None

    def getNormals(self,asNumPy=True):
        """Estimates the normals for reach point, returns as NumPy Array if True"""
        # self.pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        if self.point_cloud_loaded:
            if asNumPy:
                return np.asarray(self.pcd.normals)
            else:
                return self.pcd.normals
        else:
            print("No Point Cloud Loaded, Cannot Get Normals")
            return None

    def test(self):
        # Open the Point Cloud
        self.plyOpen()

        # Visualize the Point Cloud
        self.visPointCloud()

        # Get a NumPy Array
        npd = self.getNumPyData()
        print(npd)

        # Get Normals
        norms = self.getNormals(True)
        print(norms)

        return None

if __name__ == '__main__':
    try:
        print("Testing Open3D")
        myOpen3d = WrapperOpen3d("../data/LeafPointCloud.ply")
        myOpen3d.test()
    except:
        print("ERROR, EXCEPTION THROWN")
