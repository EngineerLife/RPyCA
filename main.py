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
from helperFiles.csvFiles import *

# function to run PCA and RPCA
def runAnalysis(X, lamScale):
    # SVD PCA
#    u, s, vh = np.linalg.svd(X)
#    print("PCA thru SVD Sigma matrix: ",s)

    maxRank = np.linalg.matrix_rank(X)
#    print("Max Rank: ", maxRank)

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
    alpha = 0.7

#    print((1/math.sqrt(max(T.shape[0],T.shape[1]))))
#    newLambda = (1/math.sqrt(max(T.shape[0],T.shape[1])))* lamScale
#    print("Norm lambda: ", newLambda/lamScale)

    print("LAMBDA SCALE WANT: ", lamScale/(1/math.sqrt(max(T.shape[0],T.shape[1]))))

    [U, E, VT, S, B] = sRPCA(T.shape[0], T.shape[1], u, v, vecM, vecEpsilon, maxRank, lam=lamScale)

    S = S.todense()    # keep
    E = np.diag(E)
#    print("Dense S: ", S)   # eh
    ue = np.dot(U, E)
    L = np.dot(ue, VT)

    print("OG SHAPES, X U E VT L: ", X.shape, U.shape, E.shape, VT.shape, L.shape)

    # TODO
    # L^hat = u^hat dot E^hat dot VT^hat
    hatRows = len(E[E > 0])
    Uhat = U[:,:hatRows]
    Ehat = np.diag(E[E > 0])    # sigma^hat old note: (set to threshold and set them to 0)
    VThat = VT[:hatRows]
    VTta = VT[hatRows:]

    print(Uhat.shape, Ehat.shape, VThat.shape, VTta.shape)

#    print("# non-zero values: " + str(len(np.where(E>0)[0])))
#    print("# negative values: " + str(len(np.where(E<0)[0])))
#    print("# non-zero values: " + str(E[E>0]))

#    XVTT = np.dot(X,VT.T)
#    S1 = np.dot(XVTT, VT)
#    print(S.shape, S1.shape)
#    print(S, "\n\n", S1)
#    exit(0)

#    Xls = L + S
#    Xt = X.T
    # NOTE KEEP THESE 2 LINES
#    xtx = np.dot(Xt, X)
# TODO email haitao:
#    print(xtx.shape)
#    exit(0)
    
    warnings.filterwarnings('always')
    print("\nL mean val: \n", np.mean(L))
    print("S max val: ", S.max(), "   difference: ", S.max()/np.mean(L), "\n")
    if abs(np.mean(L)) < 0.001: # arbitrary value
        print(L)
        logMsg(2, "L matrix seems to be empty.")
        warnings.warn('L matrix seems to be empty.\n')
    if abs(S.max()/np.mean(L)) < 0.01:  # arbitrary value
        print(S)
        logMsg(2, "S matrix seems to be empty.")
        warnings.warn('S matrix seems to be empty.\n')
    warnings.filterwarnings('ignore')

#    return S, X, s, E, L, maxRank
    return S, L, VTta, maxRank

def preproc(filename, l, alpha, typ):
    # get X matrix then one-hot encode columns & creates final matrix
    if typ == "p":
        X = getData(filename)   # loads and formats data from file
        newX = createMatrixProposal(X)  # This creates the matrix according to the OG Kathleen paper
    else:
        X = thesisDataset()
#        print("preproc: ", X.shape)
        newX = createMatrix(X)  # main thesis dataset (default)

    X = np.asmatrix(newX)
    X.astype(float)     #TODO investigate this??? should be int? float? is it needed?
#    X = cleanMat(X)    # TODO do we even need this too?
    X = normMat(X)
    print("Finished pre-processing. Running analysis...")
    return X

def thesisDataset():
    # NOTE UNB INTRUSION DETECTION DATASET
#    days = ['Monday','Tuesday','Wednesday','Thursday','Friday']
    files = 'thurs'
    testing = getFile(files, True)
    testing = [testing[1]]
#    print(testing)
    # loads and formats data from file
    return loadFile(testing)   # TODO this is only for getting the Thurs morning file for journal

# float range function
def frange(start, stop, step):
    i = start
    while i < stop:
        yield i
        i += step


# !!!!!!TODO make the input for creating X normalized (csv or something)


# main function
if __name__ == '__main__':
    setLog("log3")
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
        malPkts1, malPkts2, malPkts3 = listLabels("phase-all-MORE-counts.txt")
        # puts all malicious packet lists into one
        mpc = np.concatenate((malPkts1, malPkts2, malPkts3))
    else:
#        mpc = 'TrafficLabelling/Thursday-WorkingHours-Morning-SHORT-WebAttacks.pcap_ISCX.csv'
        mpc = 'TrafficLabelling/Thursday-WorkingHours-Morning-8100-SHORT-WebAttacks.pcap_ISCX.csv'

    i = 1
    alpha = 0.7
    # 0.027 is so far best lambda (I think)
    # so 0.554 is the best lambdaScale (I think)


#    fig = plt.figure()
#    fig.subplots_adjust(left=0.2, bottom=0.05, right=0.8, hspace=0.5, wspace=0.6)


    # NOTE 0.05 got f1 score of 1 for rf!!
    for l in lam:
#    for l in frange(0.04, 0.06, 0.01): #lam:
        logMsg(1, " Next Lambda: %s" % (str(lam)))
        print("\n\nNEXT LAMBDA: ", l)
        X = preproc("inside/LLS_DDOS_2.0.2-inside-all-MORE", l, alpha, typ)
#        print("X: ",X.shape)

        # TODO 
        splitOn = [4050, 3050]
        # X1 = train, X2 = test, X3 = validate
        X1, X2 = np.split(X, [splitOn[0]])
        X2, X3 = np.split(X2, [splitOn[1]])

        # runs RPCA
        S1, L1, VTta, maxRank = runAnalysis(X1, l)
        # test
#        S2, X2, E2, L2, maxRank1 = runAnalysis(X, l)
        # TODO X2 dot V1 dot V1.T # V's are slimmed
        X2VTT = np.dot(X2, VTta.T)
        S2 = np.dot(X2VTT, VTta)
        L2 = X2 - S2
        print("X2 SHAPES: X L S", X2.shape, L2.shape, S2.shape)

        # validate
        # TODO X3 dot V1 dot V1.T
        X3VTT = np.dot(X3, VTta.T)
        S3 = np.dot(X3VTT, VTta)
        L3 = X3 - S3
        print("X3 SHAPES: X L S", X3.shape, L3.shape, S3.shape)

        '''
        # phase 1
        S1, X1, s1, E1, maxRank1 = preproc("inside/LLS_DDOS_2.0.2-inside-phase-1", l, alpha, typ)

        # phase 2
        S2, X2, s2, E2, maxRank2 = preproc("inside/LLS_DDOS_2.0.2-inside-phase-2", l, alpha, typ)

        # phase 3
        S3, X3, s3, E3, maxRank3 = preproc("inside/LLS_DDOS_2.0.2-inside-phase-3", l, alpha, typ)

        x1 = fig.add_subplot(3, len(lam), i)
        x2 = fig.add_subplot(3, len(lam), i+len(lam))
        x3 = fig.add_subplot(3, len(lam), i+(len(lam)*2))

        plotter(S1,malPkts1,alpha,xname="Phase 1 w/ Lambda: "+str(l),bx=x1)
        plotter(S2,malPkts2,alpha,xname="Phase 2 w/ Lambda: "+str(l),bx=x2)
        plotter(S3,malPkts3,alpha,xname="Phase 3 w/ Lambda: "+str(l),bx=x3)
        '''
#        plotter(S1,mpc,alpha,xname="All Phases w/ LambdaScale: "+str(lam),bx=x1)
#        i += 1

        # ML/AI
        # old was 290 for "all" file
        if typ == "p":
            splitOn = 8000
        else:
            splitOn = [4050, 3050]
#            splitOn = 4050
#            splitOn = 900

        Xmat = [X1 ,X2, X3]
        Lmat = [L1, L2, L3]
        Smat = [S1, S2, S3]
        
        toRun = ["svm"]
        # TODO ask about this?????
#        runModels(X_mats, L1, S1, mpc, splitOn)
#        runModels(Xmat, Lmat, Smat, mpc, splitOn)    # NOTE this runs all models!!!
        runModels(Xmat, Lmat, Smat, mpc, splitOn, code=toRun)


