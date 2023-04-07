cimport numpy as np
import numpy as np

# Sum values from i to j inclusive
cdef inline float sumValues(const float[:] view, int i, int j):
    cdef float res = 0.
    #print(str(i) + " " + str(j))
    while i <= j:
        res += view[i]
        i += 1
    return res

def computeShortestInterval(np.ndarray[np.float32_t] series, np.ndarray[np.int32_t] layers, float fraction) -> tuple[int, int]:
    """ Compute the actual shortest interval. Params : 
        - series : a numpy array, from pandas Series to_numpy (with energy per layer)
        - layers : a numpy array, from pandas Series index to_numpy : layer numbers matching series values (same length as series) """
    cdef float totalEnergyFraction = fraction*np.sum(series)
    cdef int seriesLength = len(series)

    cdef const float [:] series_view = series
    cdef const int [:] layers_view = layers

    cdef int bestInterval_min = layers_view[0]
    cdef int bestInterval_max = layers_view[-1]
    cdef int j = 0, i_layer, j_layer
    
    for i in range(seriesLength):
        #print("i="+str(i))
        i_layer = layers_view[i]
        if j < i:
            j = i
        
        while sumValues(series_view, i, j) < totalEnergyFraction:
            j += 1
            if j >= seriesLength: # Impossible to find a covering interval at this stage
                return (bestInterval_min, bestInterval_max)
            
        j_layer = layers_view[j]
        if j_layer-i_layer < bestInterval_max - bestInterval_min:
            bestInterval_min = i_layer
            bestInterval_max = j_layer
    
    return (bestInterval_min, bestInterval_max)