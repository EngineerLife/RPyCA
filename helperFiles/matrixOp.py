import math
import numpy as np
import pandas as pd
from random import shuffle, randint, seed
from .logger import *
from .fileHandler import *

def preproc(fileName, labelsName, sample, rSeed, ratioTrain, ratioValid, oneHot=[], skip=[]):
    # Get random seed
    rSeed = getRand(rSeed)
    # Load in data from csv file
    X, featLabels, y = load(fileName, labelsName, sample, rSeed, skip)

    # ***************************************************************************************************
    # NOTE These following commands up to ****** are custom for UNB data set!!! 
    # Set y labels to 0 and 1 values
    y = pd.Series(np.where(y.values == 'BENIGN', 0, 1), y.index)
    # TODO come up with better method later for ports
    # XXX could group ports by the thousands place?
    X['SourcePort'][X['SourcePort'] >= 1024] = 1024
    X['DestinationPort'][X['DestinationPort'] >= 1024] = 1024
    # Splits Destination IP
    X[['DestIP_Byte1','DestIP_Byte2','DestIP_Byte3','DestIP_Byte4']] = X.DestinationIP.str.split(".", expand=True)
    X = X.drop(columns=['DestinationIP'])
    # one hot encodes specific columns
    X = pd.get_dummies(X, columns=['SourcePort', 'DestinationPort', 'Protocol', 'DestIP_Byte1','DestIP_Byte2','DestIP_Byte3','DestIP_Byte4'])
    # ***************************************************************************************************
   
    # Turn X into Numpy matrix
    Xnp = np.asmatrix(X.to_numpy(), dtype=float)
    # Turn y into Numpy array
    ynp = np.asarray(y.to_numpy())
    # Normalize columns in Xnp
    Xnp = normMat(Xnp)

#    X, fls = createMatrix(X, preOp, featLabels)  # main thesis dataset (default)
    print("X SHAPE:", Xnp.shape)
    return randData(Xnp, y, rSeed, ratioTrain, ratioValid)

def preprocLLSDOS(fileName, labelsName, rSeed, ratioTrain, ratioValid, oneHot=[], skip=[]):
    # Load in data from csv file
    X, featLabels, y = load(fileName, labelsName, skip)

    # ***************************************************************************************************
    # NOTE These following commands up to ****** are custom for LLS_DOS data set!!! 
    X['SourcePort'][np.isnan(X['SourcePort'])] = -1
    X['DestinationPort'][np.isnan(X['DestinationPort'])] = -1
    # add 2 new features to dataframe: port # >= 1024 for src and dest.
    portRangeSrc = pd.Series(np.where(X['SourcePort'] >= 1024, 1, 0), X.index)
    portRangeDst = pd.Series(np.where(X['DestinationPort'] >= 1024, 1, 0), X.index)
    X['portRangeSrc'] = portRangeSrc
    X['portRangeDst'] = portRangeDst
    # Set unimportant port numbers to 1024
    X['SourcePort'][X['SourcePort'] >= 1024] = 1024
    X['DestinationPort'][X['DestinationPort'] >= 1024] = 1024
    # Splits IP's
    X['SrcIP_Byte1'] = X.Source.str.split(".", expand=True)[0]
    X['DestIP_Byte1'] = X.Destination.str.split(".", expand=True)[0]
    X = X.drop(columns=['Destination', 'Source'])
    # one hot encodes specific columns
    X = pd.get_dummies(X, columns=['SrcIP_Byte1', 'DestIP_Byte1', 'SourcePort', 'DestinationPort', 'Protocol'])
    # ***************************************************************************************************
    
    # Turn X into Numpy matrix
    Xnp = np.asmatrix(X.to_numpy(), dtype=float)
    # Turn y into Numpy array
    ynp = np.asarray(y.to_numpy())
    # Normalize columns in Xnp
    Xnp = normMat(Xnp)

#    X, fls = createMatrix(X, preOp, featLabels)  # main thesis dataset (default)
    print("X SHAPE:", Xnp.shape)
    return randData(Xnp, y, rSeed, ratioTrain, ratioValid)


#def preprocKaggle(fileName, labelsName, rSeed, ratioTrain, ratioValid, oneHot=[], skip=[]):
    # Load in data from csv file
#    X, featLabels, y = load(fileName, labelsName, skip)
    # one hot encodes specific columns
#    X = pd.get_dummies(X, columns=['ethnicity','gender','hospital_admit_source','icu_admit_source','icu_stay_type','icu_type','apache_3j_bodysystem','apache_2_bodysystem'])
#    X = pd.get_dummies(X, columns=oneHot)
    # Turn X into Numpy matrix
#    Xnp = np.asmatrix(X.to_numpy(), dtype=float)
    # Turn y into Numpy array
#    ynp = np.asarray(y.to_numpy())
    # Normalize columns in Xnp
#    Xnp = normMat(Xnp)
#    X, fls = createMatrix(X, preOp, featLabels)  # main thesis dataset (default)
#    print("X SHAPE:", Xnp.shape)
#    return randData(Xnp, y, rSeed, ratioTrain, ratioValid)


# float range function
def frange(start, stop, step):
    ar = []
    i = start
    while i < stop:
        ar.append(i)
#        yield i
        i += step
    return ar

# DO need this; Flow Bytes/s literally has the word "Infinity" used...
def cleanMat(M):
    if np.isnan(np.sum(M)):
        M[np.isnan(M)] = 1e-5
        print("Cleaning nulls...")
    if np.isinf(np.sum(M)):
        M[np.isinf(M)] = 1e5
        print("Cleaning infs...")
    return M

# normalizes every column in the matrix from start position to end position
def normMat(M):
    # TODO later: combine columns with (only 1) or (not many 1's in a clmn of 0's)
    cleanMat(M)
    # takes std dev of columns in M
    stdDev = np.std(M,axis=0)
    stdDev[stdDev <= 0] = 1e-5
    # Z-Score
    normed = (M - np.mean(M,axis=0)) / stdDev
#    save(normed, "normedX")
    return normed


def getRand(randSeed):
    # check if we want to randomize
    if not randSeed == -1:
        # set random seed
        if not randSeed:
            rSeed = randint(0,1000000)  # arbitrary range
        else:
            rSeed = randSeed
    else:
        rSeed = -1
    logMsg(0, "Random seed = %d" % rSeed)
    seed(rSeed)
    return rSeed


# Used for randData function
# !!! Must be global var bc func is recursive
loopError = 0

# randomizes the data in the main X matrix and cooresponding y labels
def randData(X_data, y_data, randSeed, ratioTrain=(2/3), ratioValid=(2/3)):
    randX, randy = [], []   # made this var before I realized it's your name; Randy P.

    # determine size of train, test, and validate matrices
    numItems = X_data.shape[0]
    numTrain = math.ceil(X_data.shape[0] * ratioTrain)
    numValid = math.ceil((numItems-numTrain) * ratioValid)
    # rest of data is validation
    
    # check if we want to randomize
    if not randSeed == -1:
        # set random seed
#        if not randSeed:
#            rSeed = randint(0,numItems)
#        else:
#            rSeed = randSeed
#        logMsg(0, "Random seed = %d" % rSeed)
#        seed(rSeed)    

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
    else:
        # DOESN'T randomize. Because we (hopefully) didn't want to
        #   Still have to do this for loop for consistency with below methods
        X_data = np.array(X_data)
        for index in range(numItems):
            # for X:
            randX.append(X_data[index])
            # for y:
            randy.append(y_data[index])

    # separate sections of data
    X_train = np.matrix(randX[:numTrain])
    X_test = np.matrix(randX[numTrain:(numTrain+numValid)])
    X_valid = np.matrix(randX[(numTrain+numValid):])

    y_train = np.array(randy[:numTrain])
    y_test = np.array(randy[numTrain:(numTrain+numValid)])
    y_valid = np.array(randy[(numTrain+numValid):])

    # log sizes of y labels    
    logMsg(0, "X_train size: %s" % str(X_train.shape))
    logMsg(0, "X_valid size: %s" % str(X_valid.shape))
    logMsg(0, "X_test size: %s" % str(X_test.shape))
    logMsg(0, "y_train class counts: %s" % str(np.unique(y_train, return_counts=True)))
    logMsg(0, "y_valid class counts: %s" % str(np.unique(y_valid, return_counts=True)))
    logMsg(0, "y_test class counts: %s" % str(np.unique(y_test, return_counts=True)))

    # check that each section has at least 2 classes
    if (not 1 in y_train) or (not 1 in y_valid) or (not 1 in y_test):
        global loopError 
        loopError += 1
        if loopError >= 10:
            print("ERROR: cannot create matrix sections! Only found 1 class.")
            logMsg(4, "Cannot create matrix sections! Only found 1 class.")  # issue if this occurs
            exit(1)
        logMsg(2, "Check failed for creating matrix sections! Revaluating...")
        return randData(X_data, y_data, ratioTrain, ratioValid)
    
    loopError = 0
    logMsg(1, "Randomizaiton of X and y matrices complete.")
    
    return [X_train, X_valid, X_test], [y_train, y_valid, y_test]
