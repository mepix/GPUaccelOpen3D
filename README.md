# GPUaccelOpen3D

This repository uses CUDA to accelerate Point Cloud Operations.

## Random Thoughts

### K-Nearest Neighbors (KNN)

Automated clustering is best performed with a DBSCAN algorithm. For the purposes of this project, I will consider two point clouds. The first point cloud will be labeled with the DBSCAN algorithm. The second point cloud will contain the same scene, from a different angle and will be unlabeled. The KNN algorithm will then label each new point.



### RANSAC

## References

- [Point Cloud Library Documentation](http://www.open3d.org/docs/release/index.html)
- [Numba Documentation](https://numba.readthedocs.io/en/stable/cuda/examples.html#matrix-multiplication)
- [Wikipedia: KNN Algorithm](https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm)
- [Wikipedia: DBSCAN Algorithm](https://en.wikipedia.org/wiki/DBSCAN)
- [Nvidia: 7 Things to know about Numba](https://developer.nvidia.com/blog/seven-things-numba/)
- [Numba Supported NumPy Features](https://numba.pydata.org/numba-doc/latest/reference/numpysupported.html)
- [Numba Supported CUDA Features](https://numba.readthedocs.io/en/stable/cuda/cudapysupported.html#numpy-support)
- [Including Code Blocks in LaTeX](https://www.overleaf.com/learn/latex/Code_listing)
