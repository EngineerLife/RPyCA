from dimredu.eRPCAviaADMMFast import eRPCA as eRPCASparse
import numpy as np


def denseToSparse(M, E):
    assert M.shape == E.shape, 'shape mismatch'
    m = M.shape[0]
    n = M.shape[1]

    u = np.empty([m * n])
    v = np.empty([m * n])
    vecM = np.empty([m * n])
    vecE = np.empty([m * n])

    k = 0
    for i in range(m):
        for j in range(n):
            u[k] = i
            v[k] = j
            vecM[k] = M[i, j]
            vecE[k] = E[i, j]
            k += 1

    return m, n, u, v, vecM, vecE


def eRPCA(M, E, **kw):
    m, n, u, v, vecM, vecE = denseToSparse(M, E)
    maxRank = np.min(M.shape)
    return eRPCASparse(m, n, u, v, vecM, vecE, maxRank, **kw)


def test_small():
    X = np.random.random(size=[5, 15])
    E = np.ones(X.shape)*1e-6
    eRPCA(X, E)


if __name__ == '__main__':
    test_small()
