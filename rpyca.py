# File created by Marissa Bennett on 02/28/2020

import warnings
import numpy as np
from helperFiles.sRPCAviaADMMFast import *
from helperFiles.logger import *

# Thoughts on how this would work:
#
# import rpyca as rp
# rp.project(X,T)           # runs projection on validation/testing set
# rp.rpca(X,V)               # runs rpca on X_train and Validation

# Global variables for use in RPCA
VTH = None   # matrix
L = None       # matrix
S = None       # matrix

##
# Makes a projection on matrix
# X1 is projected onto X2
#
# This function can be run separately from rpca,
#   AS LONG AS rpca has been ran ONCE
##
def project(X1, X2):
    # get variables
    X1 = np.asmatrix(X1).astype(float)
    X2 = np.asmatrix(X2).astype(float)
    global VTH, S, L
    VThat = VTH
    L1 = L
    S1 = S

    # Projection;   Equation: L2 = X2*(V)^hat*(VT)^hat
    X2VTT = np.dot(X2, VThat.T)
    L2 = np.dot(X2VTT, VThat)
    S2 = X2 - L2
    
    LS1 = np.concatenate((L1, S1), axis=1)
    XLS1 = np.concatenate((X1, L1, S1), axis=1)
    
    LS2 = np.concatenate((L2, S2), axis=1)
    XLS2 = np.concatenate((X2, L2, S2), axis=1)
   
    logMsg(0, "PROJ. STEPS: VThat.T: %s X2VTT: %s  L2: %s  S2: %s" % (str(VThat.T.shape), str(X2VTT.shape), str(L2.shape), str(S2.shape)))

    return [LS1, LS2], [XLS1, XLS2]
    
# Run RPCA on data
def rpca(X1, X2, l):
    X1 = np.asmatrix(X1).astype(float)
    X2 = np.asmatrix(X2).astype(float)    
    l = float(l)
    # NOTE X1 and X2 can NOT contain NaN's nor INF's!!!!!
#    cleanMat(X1)
#    cleanMat(X2)
#    S1, L1, VThat = runAnalysis(X1, l)
    runAnalysis(X1, l)
    # modify function to print
    #logMsg(0, "X1 SHAPES: X: %s  L: %s  S: %s" % (str(X1.shape), str(L1.shape), str(S1.shape)))
    # save VThat, L, and S matricies to this class
#    global VTH, S, L
#    VTH = VThat
#    L = L1
#    S = S1
    return project(X1, X2)


# function to run PCA and RPCA
def runAnalysis(X, lam):
    # for max rank, X needs to be all floats
    X = np.asmatrix(X).astype(float)

    maxRank = np.linalg.matrix_rank(X)
    logMsg(1,"Max Rank: %s" % str(maxRank))
    u, v, vecM, vecEpsilon = [], [], [], []

    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            u.append(i)
            v.append(j)
            vecEpsilon.append(1e-5)     # NOTE original value is 1e-5
            Mij = float(X[i,j])
            vecM.append(Mij)

    u = np.array(u)
    v = np.array(v)
    vecM = np.array(vecM)
    vecEpsilon = np.array(vecEpsilon)

    # B is not used in our case, but needs to be stored in order to run sRPCA
    [U, E, VT, S, B] = sRPCA(X.shape[0], X.shape[1], u, v, vecM, vecEpsilon, maxRank, lam=lam, maxIteration=250, verbose=False)

    S = S.todense()
    # Calculate L matrix
    E = np.diag(E)
    ue = np.dot(U, E)
    L = np.dot(ue, VT)

    # Calculates projector 
    # L = u^hat dot E^hat dot VT^hat
    hatRows = len(E[E > 0])
    Uhat = U[:,:hatRows]
    Ehat = np.diag(E[E > 0])
    VThat = VT[:hatRows]
    VTta = VT[hatRows:]

    logMsg(1,"# of significant features: %s out of %s" % (str(hatRows), str(X.shape[1])))

    # For Marissa's debugging:
    print("HAT SHAPES, Uhat: %s  Ehat: %s  VThat: %s  VTta: %s" % (str(Uhat.shape), str(Ehat.shape), str(VThat.shape), str(VTta.shape)))
    
    warnings.filterwarnings('always')
    if abs(np.mean(L)) < 0.001: # arbitrary value
        warnings.warn('L matrix seems to be empty.\n')
    if abs(S.max()/np.mean(L)) < 0.01:  # arbitrary value
        warnings.warn('S matrix seems to be empty.\n')
    warnings.filterwarnings('ignore')

    # save VThat, L, and S matricies to this class
    global VTH, S, L
    VTH = VThat
    L = L
    S = S

#    return S, L, VThat
    



