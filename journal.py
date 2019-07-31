# Python3 file
# Created by Marissa Bennett

import math, sys, csv, ast
import numpy as np
import matplotlib.pyplot as plt
import csvFiles
from collections import Counter
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from sRPCAviaADMMFast import *
from oneHot import *

num_rows = 5000   # TODO check to make sure attacks are in data For Thurs short file: 82840
num_feat = 323    # TODO this will change with one-hot....

# loads a list of files and extracts/forms contents
def loadFile(names):
    mat = []
    for name in names:
        count = 0
        with open(name) as fin:
#            print(len(fin.readlines()))    # counts line in file
            for line in fin:

                if count >= 1 and count <= num_rows:
                    lineData = line.split(",")

                    indexes = [1,2,3,4,5,46]
                    temp = []

                    for item in indexes:
                        temp.append(lineData[item])
                        if item == 46:
                            mat.append(temp)
                            temp = []
                elif count > num_rows:
                    newMat = np.matrix(mat)
                    #print(newMat.shape)
                    break

                count += 1
        break   # XXX this only allows for 1 file to be read
    return newMat

# prints stats on file
def printStats(filename):
    c, mal, m, a, ogMal = 0, 0, 0, 0, 0
    sourceIps, malSourceIps, malTime = [], [], []

    with open(filename) as fin:
        for line in fin:
            lineInfo = line.split(",")
#            print(lineInfo)
            # label
            if not "BENIGN" in lineInfo[84]:
                if not lineInfo[1] in malSourceIps:
                    malSourceIps.append(lineInfo[1])
                mal += 1

            # source IP
            if not lineInfo[1] in sourceIps:
                sourceIps.append(lineInfo[1])

            # timestamp
            if c == 0:  # skips header row
                c += 1
                ogMal = mal
                continue
            time = lineInfo[6].split(" ")[1]
            hour = int(time[:2].replace(":", ""))
            if hour >= 8 and hour < 12:
                timeFrame1 = "Morning"
                m += 1
                if mal > ogMal and not timeFrame1 in malTime:
                    malTime.append(timeFrame1)
            elif hour >= 1 and hour <= 5:
                timeFrame2 = "Afternoon"
                a += 1
                if mal > ogMal and not timeFrame2 in malTime:
                    malTime.append(timeFrame2)

            c += 1
            ogMal = mal

    sortedSource = sorted(sourceIps)
    dictSource = dict(Counter(sortedSource))
    sortedMalSource = sorted(malSourceIps)
    dictMalSource = dict(Counter(sortedMalSource))

    # All totals subtract the label line
    print("Total # packets:", c-1)
    print("Total # malicious packets:", mal-1)
    print("Percent of malicious to total", ((mal-1)/(c-1)*100))
    print("# Unique Source IP's:", len(dictSource)-1)
    print("# Unique Malicious Source IP's:", len(dictMalSource)-1)
    print("Malicious packets during:", malTime)
    print("# packets during", timeFrame1, "=", m)
    print("# packets during", timeFrame2, "=", a)



# function to run PCA and RPCA and graph them
def runAnalysis(X):
    # SVD PCA
    u, s, vh = np.linalg.svd(X)
    print("PCA thru SVD Sigma matrix: ")
    print(s)

    # TODO make better
    # take sub section(s) of matrix to get rank [8:15] and [3:10]
#    sub1 = X[8:15][3:10]
#    maxRank = np.linalg.matrix_rank(sub1)
    maxRank = np.linalg.matrix_rank(X)
#    maxRank = 77
    print("Max Rank: ", maxRank)
   
    T = np.asmatrix(X)  # gets shape of X
    u = []
    v = []
    vecM = []
    vecEpsilon = []
    UOrig = []
    VOrig = []
#    UOrig = np.matrix(np.random.random(size=[num_rows, maxRank]))   # ideally, U would be num_rows x 523 (if have 523 features)
#    VOrig = np.matrix(np.random.random(size=[num_feat, maxRank]))   # ideally, V would be square (523 x 523)

    # fill in data from X
    UOrig = np.matrix([[X[x][y] for y in range(maxRank)] for x in range(T.shape[0])])
    VOrig = np.matrix([[X[x][y] for y in range(maxRank)] for x in range(T.shape[1])])

#    print(UOrig.shape, VOrig.shape)

    for i in range(T.shape[0]):
        for j in range(T.shape[1]):
#            u.append(X[i][14]) # 0 -> 77, next # 77 times, 1000 times     (uses clmn 14)
#            v.append(X[14][j]) # 0..77 -> once, then again 0..77, 1000 times     (uses row 14)
            # u and v are the indices for vecM in sRPCA???
            u.append(i)
            v.append(j)
            vecEpsilon.append(1e-5)
            Mij = float(UOrig[i, :] * (VOrig.T)[:, j])
            vecM.append(Mij)

    u = np.array(u)
    v = np.array(v)
    vecM = np.array(vecM)
    vecEpsilon = np.array(vecEpsilon)
    
#    print(u.shape,v.shape)  # both should be (77000,)
#    print(vecM)
#    print(vecM.shape)
       
    [U, E, VT, S, B] = sRPCA(num_rows, num_feat, u, v, vecM, vecEpsilon, maxRank)

    print("sRPCA E (sigma) matrix: ")
    print(E)

    S = S.todense()    # keep

    print("Dense S: ", S)   # eh

    S[S < 0] = 0    # keep???
    print("# non-zero values: " + str(len(np.where(S>0)[0])) + " out of " + str(T.shape[0]*T.shape[1]))    # keep
#    plotMat(S)     # keep

    plotS(s, E, X, maxRank)


# plots matrices
def plotMat(mat):
    print("Plotting...")

    plt.matshow(mat)
#    plt.imshow(mat)
#    plt.colorbar()
    plt.show()


# plots Sigma matrices from PCA (SVD) and sRPCA
def plotS(svd, srpca, T, maxRank):
    print("Plotting...")

    T = np.asmatrix(T)
    plt.plot(range(T.shape[1]), svd, 'rs', range(maxRank), srpca, 'bo')
    plt.show()

# normalizes every column in the matrix from start position to end position
def normMat(M):

    std = np.std(M,axis=0)
#    print("# <= 0 val: " + str(len(np.where(std<=0)[0])))

    # ensures no values are 0
    stdDev = []
    for i in range(M.shape[1]):
        if std[0,i] <= 0:
            stdDev.append(1e-5)
        else:
            stdDev.append(std[0,i])
    stdDev = np.asmatrix(stdDev)

    # Z-Score
    normed = (M - np.mean(M,axis=0)) / stdDev

    # clears up matrix for exporting to csv
    nn = []
    for row in range(M.shape[0]):
        inar = []
        for clmn in range(M.shape[1]):
            inar.append(normed[row,clmn])
        nn.append(inar)

    with open('normedJournal.csv', 'w') as writeFile:
        writer = csv.writer(writeFile)
        writer.writerows(nn)
    
    writeFile.close()

    return nn

# cleans the numpy matrix of any INF or NaN values
# TODO change later so values are NOT removed
def cleanMat(M):
    if np.isnan(np.sum(X)):
        M = M[~np.isnan(X)] # just remove nan elements from vector
        print("Cleaning nulls...")
    if np.isinf(np.sum(X)):
        M = M[~np.isinf(X)] # just remove inf elements from vector
        print("Cleaning infs...")
    return M


# main func
if __name__ == '__main__':
    files = ''
    stats = False
    numSys = len(sys.argv)
    # get user input of command
    if numSys > 3:
        print("Too many args given! Please use the format:\n\n" \
              "python3 main.py <files=all> <stats=False>\n\n" \
              "Where files defaults to all, or choose <day of week>\n" \
              "      stats defaults to False, or set to True to show stats\n" \
              "Ex: python3 main.py\n" \
              "    python3 main.py Wed True\n"\
              "    python3 main.py True\n")
        exit(0)
    elif numSys == 2:
        if sys.argv[1].lower() in "True".lower():
            stats = True
    elif numSys == 3:
        if sys.argv[2].lower() in "True".lower():
            stats = True
    if numSys <= 3 and numSys > 1:
        days = ['Monday','Tuesday','Wednesday','Thursday','Friday']
        for d in days:
            if sys.argv[1].lower() in d.lower():
                files = sys.argv[1]
                break
        

    # gets label data
    testing = csvFiles.getFile(files, True)

    testing = [testing[1]]
    # loads and formats data from file
    X = loadFile(testing)   # TODO this is only for getting the Thurs morning file for journal

    # one-hot encode columns

    # Source IP
    sip = X[:,0].T
    siph = []
    # NOTE only uses the first byte of IP address
    for r in range(X.shape[0]):
        adr = sip[0,r]
        newAdr = adr.split(".")[0]
        siph.append(newAdr)
    sip = np.array(siph, dtype=int)
#    ds = dict(Counter(sip))
#    print(ds)
#    print(len(ds))
    sipOH = oneHot(sip)

    # Source Port
    sp = X[:,1].T
    sph = []
    for r in range(X.shape[0]):
        sph.append(sp[0,r])
    sp = np.array(sph, dtype=int)
    sp[sp > 1024] = 1024
    spOH = oneHot(sp)
    # Distingishing ports
    sp[sp < 1024] = 0
    sp[sp >= 1024] = 1
    dspOH = oneHot(sp)

    # Destination IP
    dip = X[:,2].T
    diph = []
    # NOTE only uses the first byte of IP address
    for r in range(X.shape[0]):
        adr = dip[0,r]
        newAdr = adr.split(".")[0]
        diph.append(newAdr)
    dip = np.array(diph, dtype=int)
    dipOH = oneHot(dip)

    # Destination Port
    dp = X[:,3].T
    dph = []
    for r in range(X.shape[0]):
        dph.append(dp[0,r])
    dp = np.array(dph, dtype=int)
    dp[dp > 1024] = 1024
    dpOH = oneHot(dp)
    # Distingishing ports
    dp[dp < 1024] = 0
    dp[dp >= 1024] = 1
    ddpOH = oneHot(dp)

    # Protocols
    p = X[:,4].T
    ph = []
    for r in range(X.shape[0]):
        ph.append(p[0,r])
    p = np.array(ph, dtype=int)
    pOH = oneHot(p)

    # Packet Length Mean
    plm = X[:,5]
    plmh = []
    for r in range(X.shape[0]):
        plmh.append(float(plm[r,0]))
    plm = np.array(plmh, dtype=int)
    plmd = plm.shape[0]
    plm = np.reshape(plm, (plmd,1))


    # creates new X matrix
#    print(sipOH.shape, spOH.shape, dipOH.shape, dpOH.shape, pOH.shape, dspOH.shape, ddpOH.shape, plm.shape)
    newX = np.concatenate((sipOH, spOH, dipOH, dpOH, pOH, dspOH, ddpOH, plm), axis=1)
#    newX = np.concatenate((sipOH, spOH, dipOH, dpOH, pOH, dspOH, ddpOH), axis=1)
    X = np.asmatrix(newX)
    X.astype(int)
    print(X.shape,X.dtype)

    # clean and normalize matrix
    X = cleanMat(X)
    X = normMat(X)

    if stats:
        printStats(testing[0])   # can only handle 1 file at a time right now

    runAnalysis(X)
