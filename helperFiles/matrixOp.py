import math, sys, ast
import numpy as np
from numpy import argmax
from random import shuffle, randint, seed
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from .logger import *
from .fileHandler import save

# float range function
def frange(start, stop, step):
    i = start
    while i < stop:
        yield i
        i += step

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
    save(normed, "normedX")
    return normed

# Function one hot encodes data
#   takes in column of data to encode
#   returns numpy matrix of encoded column data
def oneHot(clmn):
    # define example
    values = clmn
    # integer encode
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(values)
    # binary encode
    onehot_encoder = OneHotEncoder(categories='auto', sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
    # invert first example
    inverted = label_encoder.inverse_transform([argmax(onehot_encoded[0, :])])

    return onehot_encoded


# mapping values function
# USE:
#    X = mapper(T, np.matrix(X))
#    S = mapper(T, S)
#    print("Mapped Dense S: ", S)
def translate(value, leftMin, leftMax, rightMin, rightMax):
    # Figure out how 'wide' each range is
    leftSpan = leftMax - leftMin
    rightSpan = rightMax - rightMin

    # Convert the left range into a 0-1 range (float)
    valueScaled = float(value - leftMin) / float(leftSpan)

    # Convert the 0-1 range into a value in the right range.
    return rightMin + (valueScaled * rightSpan)

# maps the stuff
def mapper(T, S):
    # map values to 0-1
    newS, newSS = [], []#np.matrix()
    leftMax = np.amax(S)
    for row in range(T.shape[0]):
        for i in range(T.shape[1]):
    #            print(S[row,i])
            newS.append(translate(S[row,i], 0, leftMax, 0, 1))
        newSS.append(newS)
        newS = []
    return np.matrix(newSS)



# Used for randData function
# !!! Must be global var bc func is recursive
loopError = 0

# randomizes the data in the main X matrix and cooresponding y labels
def randData(X_data, y_data, ratioTrain=(2/3), ratioTest=(2/3)):
    randX, randy = [], []   # made this var before I realized it's your name; Randy P.

    # determine size of train, test, and validate matrices
    numItems = X_data.shape[0]
    numTrain = math.ceil(X_data.shape[0] * ratioTrain)    # default is 2/3 is training
#    numTest = math.ceil(numTrain * ratioTest)    # default is 2/3 of rest is testing
    numTest = numTrain
#    numValid = numItems - (numTrain+numTest)    # rest of rest is validation
    
    # set random seed
    rSeed = randint(0,numItems)
#    rSeed = 6102
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
    X_train = np.matrix(randX[:numTrain])
    X_test = np.matrix(randX[numTrain:(numTrain+numTest)])
    X_valid = np.matrix(randX[(numTrain+numTest):])
    X_mats = [X_train, X_test, X_valid]

    y_train = np.array(randy[:numTrain])
    y_test = np.array(randy[numTrain:(numTrain+numTest)])
    y_valid = np.array(randy[(numTrain+numTest):])
    y_mats = [y_train, y_test, y_valid]

    # log sizes of y labels    
    logMsg(0, "X_train size: %s" % str(X_train.shape))
    logMsg(0, "X_test size: %s" % str(X_test.shape))
    logMsg(0, "X_valid size: %s" % str(X_valid.shape))
    logMsg(0, "y_train class counts: %s" % str(np.unique(y_train, return_counts=True)))
    logMsg(0, "y_test class counts: %s" % str(np.unique(y_test, return_counts=True)))
    logMsg(0, "y_valid class counts: %s" % str(np.unique(y_valid, return_counts=True)))

    # check that each section has at least 2 classes
    if (not 1 in y_train) or (not 1 in y_test) or (not 1 in y_valid):
        global loopError 
        loopError += 1
        if loopError >= 10:
            print("ERROR: cannot create matrix sections!")
            logMsg(4, "ERROR: cannot create matrix sections!")  # major issue
            exit(1)
        logMsg(2, "Check failed for creating matrix sections! Revaluating...")
        return randData(X_data, y_data, ratioTrain, ratioTest)

    # XXX checking y_test counts of 1:
#    unique, counts = np.unique(y_test, return_counts=True)
#    print(unique[1], counts[1])
#    if not counts[1] == 5:
#        logMsg(2, "Not enough malicious packets in testing data. Revaluating...")
#        return randData(X_data, y_data, ratioTrain, ratioTest)

    loopError = 0
    logMsg(1, "Randomizaiton of X and y matrices complete.")
    return X_mats, y_mats




######################################  X MATRIX CREATION FUNCTIONS ######################################

# makes a list of features 
# useful for one-hot as # clmns vary
def makeFeat(lis, num, featName):
    for i in range(num):
        lis.append(featName)
    return lis


# ONLY USED FOR THE PROPOSAL/OG KATHLEEN PAPER
# PROPOSAL ONLY!!!!!!
# EXAMPLE: ['172.16.112.50' '172.16.113.148' 'TELNET'   '60'      '23'        '4170']
#               src ip          dest ip      protocol   len    src port     dest port
def createMatrixProposal(X):
    # Source IP
    sip = X[:,0].T
    siph = []
    # NOTE only uses the first byte of IP address
    for r in range(X.shape[0]):
        adr = sip[0,r]
        newAdr = adr.split(".")[0]
        siph.append(newAdr)
    sip = np.array(siph)#, dtype=float)
    sipOH = oneHot(sip)

    # Destination IP
    dip = X[:,1].T
    diph = []
    # NOTE only uses the first byte of IP address
    for r in range(X.shape[0]):
        adr = dip[0,r]
        newAdr = adr.split(".")[0]
        diph.append(newAdr)
    dip = np.array(diph)#, dtype=float)
    dipOH = oneHot(dip)

    # Protocols
    p = X[:,2].T
    ph = []
    for r in range(X.shape[0]):
        ph.append(p[0,r])
    p = np.array(ph)#, dtype=float)
    pOH = oneHot(p)

    # Packet Length
    plm = X[:,3]
    plmh = []
    for r in range(X.shape[0]):
        plmh.append(float(plm[r,0]))
    plm = np.array(plmh, dtype=float)
    plmd = plm.shape[0]
    plm = np.reshape(plm, (plmd,1))

    # Source Port
    sp = X[:,4].T
    sph = []
    spm = []
    for r in range(X.shape[0]):
        if sp[0,r] is None:     # handle missing port numbers
            spm.append(1)   # designate missing port
            sph.append(-1)
        else:
            spm.append(0)   # designate port available
            sph.append(sp[0,r])
    spm = np.array(spm, dtype=int)
    spmd = spm.shape[0]
    spm = np.reshape(spm, (spmd,1))
    sp = np.array(sph, dtype=int)
    sp[sp > 1024] = 1024
    spOH = oneHot(sp)

    # Destination Port
    dp = X[:,5].T
    dph = []
    dpm = []
    for r in range(X.shape[0]):
        if dp[0,r] is None:     # handle missing port numbers
            dpm.append(1)   # designate missing port
            dph.append(-1)
        else:
            dpm.append(0)   # designate port available
            dph.append(dp[0,r])
    dpm = np.array(dpm, dtype=int)
    dpmd = dpm.shape[0]
    dpm = np.reshape(dpm, (dpmd,1))
    dp = np.array(dph, dtype=int)
    dp[dp > 1024] = 1024
    dpOH = oneHot(dp)

    # Distingishing destination ports
    dp[dp < 1024] = 0
    dp[dp >= 1024] = 1
    ddpOH = oneHot(dp)

    # Distinguishing source ports
    sp[sp < 1024] = 0
    sp[sp >= 1024] = 1
    dspOH = oneHot(sp)

    # creates new X matrix
#    print(sipOH.shape, spOH.shape, dipOH.shape, dpOH.shape, pOH.shape, dspOH.shape, ddpOH.shape, plm.shape, spm.shape, dpm.shape)
    newX = np.concatenate((sipOH, dipOH, spOH, dpOH, dspOH, ddpOH, spm, dpm, pOH, plm), axis=1)

    # MUST UPDATE with *NEW* FEATURES (not change in one-hot amt of clmns)
    feats = []
    feats = makeFeat(feats, sipOH.shape[1], "Src IP")
    feats = makeFeat(feats, dipOH.shape[1], "Dest IP")
    feats = makeFeat(feats, spOH.shape[1], "Src Port")
    feats = makeFeat(feats, dpOH.shape[1], "Dest Port")
    feats = makeFeat(feats, dspOH.shape[1], "Dist Src Port")
    feats = makeFeat(feats, ddpOH.shape[1], "Dist Dest Port")
    feats = makeFeat(feats, spm.shape[1], "Missing Src Port")
    feats = makeFeat(feats, dpm.shape[1], "Missing Dest Port")
    feats = makeFeat(feats, pOH.shape[1], "Protocol")
    feats = makeFeat(feats, plm.shape[1], "Packet Length")
#    print(len(feats))  # prints number of features used

    return newX


# XXX notes
# specify what operation on which clmn
# need matrix
# concat new stuff on axis 1
# record which feature

# start from 1st clmn of X
def createMatrix(X, preOp,  featLabels):
    M, feats = [], []

#    print("COLUMNS TO GO THRU:",X.shape[1])
    for clmn in range(X.shape[1]):
        dataClmn = np.array(X[:,clmn].T, dtype=float)[0]
#        print("DATA:",dataClmn)
        # custom pre-operation 
        # (If only Z-transforming w/ NO 1-hot then preOp should be 0!)
        if not preOp[clmn]:
            dataClmn = np.reshape(dataClmn, (dataClmn.shape[0],1))
        # checks if clmn needs one hot encoding
        else:
            if preOp[clmn] == 2:
                dataClmn[dataClmn > 1024] = 1024    # FIXME??? Technically this does group all ports above 1024...
            dataClmn = oneHot(dataClmn)
        # add new clmn(s) to X and feature list
        if clmn == 0:
            M = dataClmn
        else:
            M = np.concatenate((M, dataClmn), axis=1)
        feats = makeFeat(feats, dataClmn.shape[1], featLabels[clmn])
    np.asmatrix(M)
    finalX = normMat(M)  # normalizes values
    return finalX, feats

