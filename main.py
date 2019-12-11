# Python3 file
# Created by Marissa Bennett

import math, sys, csv, ast, re, warnings
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.decomposition import PCA
from helperFiles.sRPCAviaADMMFast import *
from helperFiles.logger import *
from helperFiles.matrixOp import *
from helperFiles.oneHot import *
from helperFiles.fileHandler import *
from helperFiles.plotter import *
from helperFiles.models import *

import pandas as pd


# function to run PCA and RPCA
def runAnalysis(X, lamScale):
    # SVD PCA
#    u, s, vh = np.linalg.svd(X)
#    print("PCA thru SVD Sigma matrix: ",s)

    maxRank = np.linalg.matrix_rank(X)
#    print("Max Rank: ", maxRank)
    logMsg(0, "Max Rank: %d" % maxRank)
    T = np.asmatrix(X)  # gets shape of X
    u, v, vecM, vecEpsilon = [], [], [], []

    for i in range(T.shape[0]):
        for j in range(T.shape[1]):
            u.append(i)
            v.append(j)
            vecEpsilon.append(1e-5)     # NOTE original value is 1e-5
            Mij = float(T[i,j])
            vecM.append(Mij)

    u = np.array(u)
    v = np.array(v)
    vecM = np.array(vecM)
    vecEpsilon = np.array(vecEpsilon)

#    print((1/math.sqrt(max(T.shape[0],T.shape[1]))))
#    newLambda = (1/math.sqrt(max(T.shape[0],T.shape[1])))* lamScale
#    print("Norm lambda: ", newLambda/lamScale)

    scaleWant = lamScale/(1/math.sqrt(max(T.shape[0],T.shape[1])))
#    print("LAMBDA SCALE WANT: ", lamScale/(1/math.sqrt(max(T.shape[0],T.shape[1]))))
    logMsg(0, "Lambda Scale: %s" % (str(scaleWant)))

    [U, E, VT, S, B] = sRPCA(T.shape[0], T.shape[1], u, v, vecM, vecEpsilon, maxRank, lam=lamScale)

    S = S.todense()
    E = np.diag(E)
#    print("Dense S: ", S)   # eh
    ue = np.dot(U, E)
    L = np.dot(ue, VT)

#    print("OG SHAPES, X U E VT L: ", X.shape, U.shape, E.shape, VT.shape, L.shape)
    logMsg(0, "OG SHAPES, X: %s  U: %s  E: %s  VT: %s  L: %s" % (str(X.shape), str(U.shape), str(E.shape), str(VT.shape), str(L.shape)))

    # TODO
    # L^hat = u^hat dot E^hat dot VT^hat
    hatRows = len(E[E > 0])
#    print(hatRows)
    Uhat = U[:,:hatRows]
    Ehat = np.diag(E[E > 0])
    VThat = VT[:hatRows]
    VTta = VT[hatRows:]

#    UEhat = np.dot(U, Ehat)
#    Ltest = np.dot(UEhat, VT)

#    print(Uhat.shape, Ehat.shape, VThat.shape, VTta.shape)
    logMsg(0, "HAT SHAPES, Uhat: %s  Ehat: %s  VThat: %s  VTta: %s" % (str(Uhat.shape), str(Ehat.shape), str(VThat.shape), str(VTta.shape)))

# TODO email haitao:
#    xtx = np.dot(X.T, X)
#    print(xtx.shape)
    
    warnings.filterwarnings('always')
#    print("\nL mean val: \n", np.mean(L))
#    print("S max val: ", S.max(), "   difference: ", S.max()/np.mean(L), "\n")
    if abs(np.mean(L)) < 0.001: # arbitrary value
#        print(L)
        logMsg(2, "L matrix seems to be empty.")
        warnings.warn('L matrix seems to be empty.\n')
    if abs(S.max()/np.mean(L)) < 0.01:  # arbitrary value
#        print(S)
        logMsg(2, "S matrix seems to be empty.")
        warnings.warn('S matrix seems to be empty.\n')
    warnings.filterwarnings('ignore')

#    return S, X, s, E, L, maxRank
    return S, L, VThat

# TODO change parameters so that filename is not needed
def preproc(filename, typ):
    # get X matrix then one-hot encode columns & creates final matrix
    if typ == "p":
        X = getLLDOSData(filename)   # loads and formats data from file
        newX = createMatrixProposal(X)  # This creates the matrix according to the OG Kathleen paper
    else:
        X = thesisDataset()
        newX = createMatrix(X)  # main thesis dataset (default)

    X = np.asmatrix(newX)
    X.astype(float)     #TODO investigate this??? should be int? float? is it needed?
    X = normMat(X)
#    print("Finished pre-processing. Running analysis...")
    return X

def thesisDataset():
    # NOTE UNB INTRUSION DETECTION DATASET
#    days = ['Monday','Tuesday','Wednesday','Thursday','Friday']
    files = 'thurs'     # TODO remove hardcoding
    testing = getUNBFile(files, True)
    testing = [testing[1]]
#    print(testing)
    # loads and formats data from file
    return loadUNBFile(testing)   # TODO this is only for getting the Thurs morning file for journal

# float range function
def frange(start, stop, step):
    i = start
    while i < stop:
        yield i
        i += step


# !!!!!!TODO make the input for creating X normalized (csv or something)

# TODO TODO TODO:
# When explaining data:
#   take x and y data to show that using IP's as a feature makes it too easy
#   get writing done-ish before next semester
#       where chapters are complete




# main function
if __name__ == '__main__':
#    exit(0) # XXX REMOVE ME
#    setLog("logRFtestnew2")
    setLog("logCreatePlots")
#    setLog("trash")
    numSys = len(sys.argv)
    lam = []
    typ = ""
    # Gets arguments from command line 
    if numSys > 1:
        st = sys.argv[1].split(",")
        for i in st:
            if re.search("[-+]?[0-9]*\.?[0-9]+", i):
                lam.append(float(i))
        if numSys == 3 and sys.argv[2].lower() in "proposal":
            typ = "p"
    else:
        lam = [1]
    logMsg(1, " Start variables= lambda(s): %s  type: %s" % (str(lam), typ))

    # TODO incorporate file weekday name into args if main thesis

    
    # XXX create y, create X, preproc X, split X and y into 3 chunks
    # XXX Run RPCA, calculate other 2 chunks, run ML models/NN

    if typ == "p":
        # retrieves malicious packet indexes
        malPkts1, malPkts2, malPkts3 = listLLDOSLabels("phase-all-MORE-counts.txt")
        # puts all malicious packet lists into one
        mpc = np.concatenate((malPkts1, malPkts2, malPkts3))
    else:
#        mpc = 'datasets/TrafficLabelling/Thursday-WorkingHours-Morning-SHORT-WebAttacks.pcap_ISCX.csv'
        mpc = 'datasets/TrafficLabelling/Thursday-WorkingHours-Morning-8100-SHORT-WebAttacks.pcap_ISCX.csv'
        y = loadUNBLabels(mpc)

    # creat X and y
    X = preproc("datasets/inside/LLS_DDOS_2.0.2-inside-all-MORE", typ)
    if typ == "p":
        y = createY(len(X), mpc)

    # randomizes data and creates separated matrices
#    [X1, X2, X3], ymat = randData(X, y)
#    [X1, X2, X3], ymat = randData(X, y, ratioTest=0.06)
    [X1, X2, X3], ymat = randData(X, y, 1/3, 1/3)
    # ratioTrain = 0.1
    
    # ML model to run
    toRun = ["svm"]
    goodLam = []    # holds good lambdas
    allData, goodData = [], []  # XXX plotting

    # for random seed: 6102 lambdas: 0.08, 0.09 .1
#    artest = [0.08] * 10
#    artest = [0.1] * 10

#    for l in lam:
    for l in frange(0.01, 0.1, 0.01):
#    for l in artest:
        logMsg(1, "Lambda: %s" % (str(l)))
        print("\n\nLAMBDA: ", l)

        # runs RPCA
        S1, L1, VThat = runAnalysis(X1, l)
#        print("X1 SHAPES: X L S", X1.shape, L1.shape, S1.shape)
        logMsg(0, "X1 SHAPES: X: %s  L: %s  S: %s" % (str(X1.shape), str(L1.shape), str(S1.shape)))

        # CHECK for S1 and L1
#        X1VTT = np.dot(X1, VThat.T)
#        LC1 = np.dot(X1VTT, VThat)
#        SC1 = X1 - LC1
#        print("L:\n%s\n%s" % (str(L1), str(LC1)))
#        print("S:\n%s\n%s" % (str(S1), str(SC1)))

        # test
#        print((VThat.T).shape)
        X2VTT = np.dot(X2, VThat.T)
#        print(X2VTT.shape)
        L2 = np.dot(X2VTT, VThat)
        S2 = X2 - L2
#        print("X2 SHAPES: X L S", X2.shape, L2.shape, S2.shape)
        logMsg(0, "X2 SHAPES: X: %s  L: %s  S: %s" % (str(X2.shape), str(L2.shape), str(S2.shape)))

        # ML/AI
        Xmat = [X1, X2]
        Lmat = [L1, L2]
        Smat = [S1, S2]
        ymatX12 = [ymat[0], ymat[1]]

#        runModels(Xmat, Lmat, Smat, mpc, splitOn)    # NOTE this runs all models!!!
        res, dall = runModels(Xmat, Lmat, Smat, ymatX12, code=toRun)
        print(dall)
        allData.append(dall)
        if res:
#            goodLam.append(l)
            print("Validating...")
            logMsg(1, "Validating GOOD Lambda: %s" % (str(l)))
            # runs RPCA
#            S1, L1, VThat = runAnalysis(X1, l)

            # validate
            X3VTT = np.dot(X3, VThat.T)
            L3 = np.dot(X3VTT, VThat)
            S3 = X3 - L3
#            print("X3 SHAPES: X L S", X3.shape, L3.shape, S3.shape)
            logMsg(0, "X3 SHAPES: X: %s  L: %s  S: %s" % (str(X3.shape), str(L3.shape), str(S3.shape)))

            # ML/AI
            Xmat = [X1, X3]
            Lmat = [L1, L3]
            Smat = [S1, S3]
            ymatX13 = [ymat[0], ymat[2]]

            res, dgood = runModels(Xmat, Lmat, Smat, ymatX13, code=toRun)
            print(dgood)
            goodData.append(dgood)

    # XXX quick plot graph
    # (each matrix data in clmn (X, LS, XLS), each run in rows)
#    df = pd.DataFrame(allData, columns=['X', 'CONCAT LS', 'CONCAT XLS'])
#    boxplot = df.boxplot(column=['X', 'CONCAT LS', 'CONCAT XLS'])
#    plt.title('Tuning F1 Scores')
#    plt.show()
#    plt.savefig('allData0.08.png')
#    plt.savefig('allData0.1.png')

    exit(0)
    df = pd.DataFrame(goodData, columns=['X', 'CONCAT LS', 'CONCAT XLS'])
    boxplot = df.boxplot(column=['X', 'CONCAT LS', 'CONCAT XLS'])
    plt.title('Validation Matrix F1 Scores')
    plt.ylabel("F1 Scores")
    plt.show()
#    plt.savefig('goodData0.08.png')
#    plt.savefig('goodData0.1.png')
    plt.savefig('final_f1_scores.png')

    # TODO use histograms
    '''
    # logs all good lambdas
    logMsg(1, "Lambda Tuning Complete")
    logMsg(0, "GOOD Lambdas: %s" % (str(goodLam)))

    for glam in goodLam:
        logMsg(1, "GOOD Lambda: %s" % (str(glam)))
        # runs RPCA
        S1, L1, VThat = runAnalysis(X1, glam)

        # validate
        X3VTT = np.dot(X3, VThat.T)
        L3 = np.dot(X3VTT, VThat)
        S3 = X3 - L3
#        print("X3 SHAPES: X L S", X3.shape, L3.shape, S3.shape)
        logMsg(0, "X3 SHAPES: X: %s  L: %s  S: %s" % (str(X3.shape), str(L3.shape), str(S3.shape)))

        # ML/AI
        Xmat = [X1, X3]
        Lmat = [L1, L3]
        Smat = [S1, S3]
        ymatX13 = [ymat[0], ymat[2]]
                                        
        runModels(Xmat, Lmat, Smat, ymatX13, code=toRun)
    '''
