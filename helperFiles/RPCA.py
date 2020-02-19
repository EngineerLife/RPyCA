import math, warnings
import numpy as np
from sklearn.decomposition import PCA
from helperFiles.sRPCAviaADMMFast import *
from helperFiles.logger import *

# function to run PCA and RPCA
def runAnalysis(X, lam):
    # SVD PCA
#    u, s, vh = np.linalg.svd(X)
#    print("PCA thru SVD Sigma matrix: ",s)

    # TODO update way of calculating Max Rank later
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

    lamScaled = lam/(1/math.sqrt(max(T.shape[0],T.shape[1])))
    logMsg(0, "Lambda Scale: %s" % (str(lamScaled)))

    # B is not used in our case, but needs to be stored in order to run sRPCA
    [U, E, VT, S, B] = sRPCA(T.shape[0], T.shape[1], u, v, vecM, vecEpsilon, maxRank, lam=lam)

    S = S.todense()
    # Calculate L matrix
    E = np.diag(E)
    ue = np.dot(U, E)
    L = np.dot(ue, VT)

    logMsg(0, "OG SHAPES, X: %s  U: %s  E: %s  VT: %s  L: %s" % (str(X.shape), str(U.shape), str(E.shape), str(VT.shape), str(L.shape)))

    # Calculates projector 
    # L^hat = u^hat dot E^hat dot VT^hat
    hatRows = len(E[E > 0])
    Uhat = U[:,:hatRows]
    Ehat = np.diag(E[E > 0])
    VThat = VT[:hatRows]
    VTta = VT[hatRows:]

    logMsg(0, "HAT SHAPES, Uhat: %s  Ehat: %s  VThat: %s  VTta: %s" % (str(Uhat.shape), str(Ehat.shape), str(VThat.shape), str(VTta.shape)))

# TODO email haitao:
#    xtx = np.dot(X.T, X)
#    print(xtx.shape)
    
    warnings.filterwarnings('always')
    if abs(np.mean(L)) < 0.001: # arbitrary value
        logMsg(2, "L matrix seems to be empty.")
        warnings.warn('L matrix seems to be empty.\n')
    if abs(S.max()/np.mean(L)) < 0.01:  # arbitrary value
        logMsg(2, "S matrix seems to be empty.")
        warnings.warn('S matrix seems to be empty.\n')
    warnings.filterwarnings('ignore')

    return S, L, VThat

