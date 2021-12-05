#!/usr/bin/env python3

import numpy as np
import open3d as o3d
import copy
import matplotlib.pyplot as plt


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

    def getClusterLabels(self):
        """Uses DBSCAN to cluster the point cloud and returns a NumPy label array"""
        if self.point_cloud_loaded:
            # Run the Clustering Algorithm
            with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
                labels = np.array(self.pcd.cluster_dbscan(eps=0.02, min_points=10, print_progress=True))

            max_label = labels.max()
            pcd = copy.deepcopy(self.pcd)
            print(f"point cloud has {max_label + 1} clusters")
            colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
            colors[labels < 0] = 0
            pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
            o3d.visualization.draw_geometries([pcd],
                                              zoom=0.455,
                                              front=[-0.4999, -0.1659, -0.8499],
                                              lookat=[2.1813, 2.0619, 2.0999],
                                              up=[0.1204, -0.9852, 0.1215])


            return labels
        else:
            print("No Point Cloud Loaded, Cannot Cluster Points")
            return None

    def drawBox(self):
        print("Let\'s draw a cubic using o3d.geometry.LineSet")
        points = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [0, 0, 1], [1, 0, 1],
                  [0, 1, 1], [1, 1, 1]]
        lines = [[0, 1], [0, 2], [1, 3], [2, 3], [4, 5], [4, 6], [5, 7], [6, 7],
                 [0, 4], [1, 5], [2, 6], [3, 7]]
        colors = [[1, 0, 0] for i in range(len(lines))]
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(points)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.colors = o3d.utility.Vector3dVector(colors)
        o3d.visualization.draw_geometries([line_set])

    def drawOrigin(self):
        print("Let\'s draw a cubic using o3d.geometry.LineSet")
        points = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]
        lines = [[0, 1], [0, 2], [0, 3]]
        colors = [[1, 0, 0] for i in range(len(lines))]
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(points)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.colors = o3d.utility.Vector3dVector(colors)
        o3d.visualization.draw_geometries([line_set,self.pcd])



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

    def convertNumPyToPointCloud(self,arr):
        """Converts a Nx9 NumPy array [X,Y,Z,R,G,B,NormX,NormY,NormZ] to Open3d"""
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(arr[:,[0,1,2]])
        pcd.colors = o3d.utility.Vector3dVector(arr[:,[3,4,5]])
        pcd.normals = o3d.utility.Vector3dVector(arr[:,[6,7,8]])
        return pcd


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

        # Convert the NumPy Data back to Point Cloud
        myOpen3d.pcd = myOpen3d.convertNumPyToPointCloud(myNumPyArray)
        myOpen3d.visPointCloud()

        # TODO: Clean this up, this is mostly scratch work
        print(myOpen3d.getClusterLabels())
        myOpen3d.drawOrigin()


    except:
        print("ERROR, EXCEPTION THROWN")
