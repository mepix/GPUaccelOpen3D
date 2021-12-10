# GPUaccelOpen3D

This repository uses CUDA to accelerate Point Cloud Operations.

## Random Thoughts

### K-Nearest Neighbors (KNN)

Automated clustering is best performed with a DBSCAN algorithm. For the purposes of this project, I will consider two point clouds. The first point cloud will be labeled with the DBSCAN algorithm. The second point cloud will contain the same scene, from a different angle and will be unlabeled. The KNN algorithm will then label each new point.

### RANSAC

## References

### Numba Documentation

- [Numba Documentation](https://numba.readthedocs.io/en/stable/cuda/examples.html#matrix-multiplication)
- [Numba Supported NumPy Features](https://numba.pydata.org/numba-doc/latest/reference/numpysupported.html)
- [Numba Supported CUDA Features](https://numba.readthedocs.io/en/stable/cuda/cudapysupported.html#numpy-support)
- [Numba CUDA Atomic Operations](https://numba.pydata.org/numba-doc/latest/cuda/intrinsics.html)
- [Numba CUDA Random Number Generators](https://numba.readthedocs.io/en/stable/cuda/random.html#a-simple-example)

### Wikipedia

- [Wikipedia: KNN Algorithm](https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm)
- [Wikipedia: DBSCAN Algorithm](https://en.wikipedia.org/wiki/DBSCAN)
- [Wikipedia Radix Sort](https://en.wikipedia.org/wiki/Radix_sort)

### General

- [Point Cloud Library Documentation](http://www.open3d.org/docs/release/index.html)
- [Nvidia: 7 Things to know about Numba](https://developer.nvidia.com/blog/seven-things-numba/)
- [Including Code Blocks in LaTeX](https://www.overleaf.com/learn/latex/Code_listing)
- [NYU: Numba CUDA Matrix Multiply](https://nyu-cds.github.io/python-numba/05-cuda/)
