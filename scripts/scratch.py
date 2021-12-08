#!/usr/bin/env python3

import numpy as np
from numba import cuda, float32, uint16, int32 # GPU Optimizations
import math


K_NEAREST = 5

def my_selection_sort():
    """
    https://www.cs.utexas.edu/~mitra/csFall2017/cs303/lectures/basic_algo.html
    FOR i = 0 to array length - 2
      SET Min to array[i]
      SET MinIndex to i
      FOR j = i + 1 to array length - 1
        IF array[j] < Min THEN
          Min = array[j]
          MinIndex = j
        ENDIF
      ENDFOR
      SET array[MinIndex] to array[i]
      SET array[i] to Min
    ENDFOR
    """

    # dummy distances
    y_train = np.array([-17,-12,-15,-13,-13,-11,-14,-15,-16]).reshape([-1,1])
    print(y_train.shape)
    print(y_train)

    distances = np.array([7,2,5,3,3,1,4,5,6]).reshape([-1,1])
    print(distances.shape)
    print(distances)

    for i in range(distances.shape[0]-1):
        min_val = distances[i,0]
        min_idx = i
        y_val = y_train[i,0]
        for j in range(i+1,distances.shape[0]):
            if distances[j,0] < min_val:
                min_val = distances[j,0]
                min_idx = j
                y_val = y_train[j,0]
        distances[min_idx,0] = distances[i,0]
        distances[i,0] = min_val

        y_train[min_idx,0] = y_train[i,0]
        y_train[i,0] = y_val

    print(distances)
    print(y_train)

    return y_train

def my_mini_histo(top_k):
    print(top_k)

    # Vote and Assign Labels
    counter = np.zeros(top_k.shape)
    count_max = 0
    val_max = 0
    for i in range(top_k.shape[0]):
        val_current = top_k[i,0]
        count_now = 0
        for j in range(top_k.shape[0]):
            if (val_current == top_k[j,0]):
                count_now += 1
                if (count_now > count_max):
                    count_max = count_now
                    val_max = top_k[j]
        counter[i] = count_now
        
    print(counter)
    print(val_max)

if __name__ == '__main__':
    k = 5
    print("RUNNING SORT")
    y_sorted = my_selection_sort()
    top_k = y_sorted[0:k]
    print(top_k)
    print("RUNNING COUNTER")
    my_mini_histo(top_k)
