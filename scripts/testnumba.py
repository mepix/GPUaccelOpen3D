#!/usr/bin/env python3

from numba import cuda
import numpy as np
import math

@cuda.jit
def increment_a_2D_array(an_array):
    x, y = cuda.grid(2)
    if x < an_array.shape[0] and y < an_array.shape[1]:
       an_array[x, y] += 1

threadsperblock = (16, 16)
an_array = np.array([[1,2,3],[4,5,6]])
print(an_array)
blockspergrid_x = math.ceil(an_array.shape[0] / threadsperblock[0])
blockspergrid_y = math.ceil(an_array.shape[1] / threadsperblock[1])
blockspergrid = (blockspergrid_x, blockspergrid_y)
print(blockspergrid)
increment_a_2D_array[blockspergrid, threadsperblock](an_array)
print(an_array)
