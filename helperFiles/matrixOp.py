import math, sys, ast
import numpy as np

# normalizes every column in the matrix from start position to end position
def normMat(M):
    # TODO later: combine columns with only 1 or not many 1's in clmn of 0's

    # takes std dev of columns in M
    stdDev = np.std(M,axis=0)
    # Z-Score
    normed = (M - np.mean(M,axis=0)) / stdDev
    return normed


# cleans the numpy matrix of any INF or NaN values
# TODO change later so values are NOT removed
# XXX MAY NOT USE THIS
def cleanMat(M):
    if np.isnan(np.sum(M)):
        M = M[~np.isnan(M)] # just remove nan elements from vector
        print("Cleaning nulls...")
    if np.isinf(np.sum(M)):
        M = M[~np.isinf(M)] # just remove inf elements from vector
        print("Cleaning infs...")
    return M
