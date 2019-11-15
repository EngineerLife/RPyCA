import math, sys, ast
import numpy as np
from random import shuffle, randint, seed
from .logger import *

# normalizes every column in the matrix from start position to end position
def normMat(M):
    # TODO later: combine columns with only 1 or not many 1's in clmn of 0's

    # takes std dev of columns in M
    stdDev = np.std(M,axis=0)
    # Z-Score
    normed = (M - np.mean(M,axis=0)) / stdDev
    return normed


# randomizes the data in the main X matrix and cooresponding y labels
def randData(X_data, y_data):
    randX, randy = [], []   # made this var before I realized it's your name; Randy P.

    # determine size of train, test, and validate matrices
    numItems = X_data.shape[0]
    numTrain = math.ceil(X_data.shape[0] * (2/3))    # 2/3 is training
    numTest = math.ceil(numTrain * (1/3))    # 1/3 of rest is testing
#    numValid = numItems - (numTrain+numTest)    # rest of rest is validation
    
    # set random seed
#    rSeed = randint(0,numItems)
    rSeed = 4322
    logMsg(0, "Random seed = %d" % rSeed)
    seed(rSeed)

    # produce a random arrangement
    order = np.arange(numItems)
    shuffle(order)

    # randomize rows indexes based on order variable
    X_data = np.array(X_data)
    for index in order:
        # for X:
        randX.append(X_data[index])
        # for y:
        randy.append(y_data[index])

    # separate sections of data
#    X_train = np.matrix(randX[:numTrain])
#    X_test = np.matrix(randX[numTrain:(numTrain+numTest)])
#    X_valid = np.matrix(randX[(numTrain+numTest):])
    X_train = np.matrix(randX[:numTrain])
    X_test = np.matrix(randX[numTrain:(numTrain+numTest-1)])
    X_test[1] = randX[numTrain]
    X_valid = np.matrix(randX[(numTrain+numTest-1):])
    X_mats = [X_train, X_test, X_valid]

    print(X_test[0] == X_test[1])

    y_test = np.array(randy[numTrain:(numTrain+numTest-1)])
    y_test[1] = 1
    print(y_test[0], y_test[1])
#    y_test = np.array(randy[numTrain:(numTrain+numTest-1)])
    y_valid = np.array(randy[(numTrain+numTest-1):])
    y_train = np.array(randy[:numTrain])
#    y_test = np.array(randy[numTrain:(numTrain+numTest)])
#    y_valid = np.array(randy[(numTrain+numTest):])
    y_mats = [y_train, y_test, y_valid]

    # check that each section has at least 2 classes
    
    print(np.unique(y_train, return_counts=True))
    print(np.unique(y_test, return_counts=True))
    print(np.unique(y_valid, return_counts=True))
    if (not 1 in y_train) or (not 1 in y_test) or (not 1 in y_valid):
        logMsg(2, "Check failed for creating matrix sections! Revaluating...")
        return randData(X_data, y_data)

    logMsg(1, "Randomizaiton of X and y matrices complete.")
    return X_mats, y_mats


# cleans the numpy matrix of any INF or NaN values
# todo change later so values are NOT removed
# XXX MAY NOT USE THIS
def cleanMat(M):
    if np.isnan(np.sum(M)):
        M = M[~np.isnan(M)] # just remove nan elements from vector
        print("Cleaning nulls...")
    if np.isinf(np.sum(M)):
        M = M[~np.isinf(M)] # just remove inf elements from vector
        print("Cleaning infs...")
    return M
