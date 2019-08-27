# Python3 file
# Created by Marissa Bennett

import math, sys, csv, ast
import numpy as np
import matplotlib.pyplot as plt
import csvFiles
from plotter import *
from collections import Counter
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from sRPCAviaADMMFast import *
from oneHot import *

''' 
dictionary that holds the feature sizes
 each next feature increments off of the previous ones
 for ease of figuring out the defining feature(s) in sRPCA
'''
feat = {
    "Src IP": 0,
    "Src Port": 0,
    "Dist Src Port": 0,
    "Dest IP": 0,
    "Dest Port": 0,
    "Dist Dest Port": 0,
    "Protocol": 0,
    "Pkt Len Mean": 0
}

# loads a list of files and extracts/forms contents
def loadFile(names):
    num_rows = 1000     # rows of data in matrix
    mat, malPkts = [], []
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
                    # create list of malicious packet indices
                    if not "BENIGN" in lineData[84]:
                        malPkts.append(count+1)
                elif count > num_rows:
                    newMat = np.matrix(mat)
                    #print(newMat.shape)
                    break

                count += 1
        break   # NOTE this only allows for 1 file to be read
    return newMat, malPkts

# prints stats on file
def printStats(filename):
    c, mal, m, a, ogMal = 0, 0, 0, 0, 0
    sourceIps, malSourceIps, malTime = [], [], []

    with open(filename) as fin:
        for line in fin:
            lineInfo = line.split(",")
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
def runAnalysis(X, l, alpha, malPkts):
    # SVD PCA
    u, s, vh = np.linalg.svd(X)
#    print("PCA thru SVD Sigma matrix: ",s)

    # TODO make better
    # take sub section(s) of matrix to get rank [8:15] and [3:10]
#    sub1 = X[:f][:rr]
#    maxRank = np.linalg.matrix_rank(sub1)
    maxRank = np.linalg.matrix_rank(X)
#    print("Max Rank: ", maxRank)

    T = np.asmatrix(X)  # gets shape of X
    u, v, vecM, vecEpsilon = [], [], [], []

    for i in range(T.shape[0]):
        for j in range(T.shape[1]):
            u.append(i)
            v.append(j)
            vecEpsilon.append(1e-5)     # NOTE original value was 1e-5
            Mij = float(T[i,j])
            vecM.append(Mij)

    u = np.array(u)
    v = np.array(v)
    vecM = np.array(vecM)
    vecEpsilon = np.array(vecEpsilon)
    

#    l = 0.015    # 0.015 good lambda for me?? NOTE
    # 0.64 good lambda
    [U, E, VT, S, B] = sRPCA(T.shape[0], T.shape[1], u, v, vecM, vecEpsilon, maxRank, lam=l)

#    test = np.dot(U,np.diag(E))
#    print(np.dot(test,VT))
#    print("sRPCA E (sigma) matrix: ",E)

    S = S.todense()    # keep
#    print("Dense S: ", S)   # eh

    S[S < 0] = 0    # keep
        
    # map values to 0-1
    newS, newSS = [], []#np.matrix()
    leftMax = np.amax(S)
    for row in range(T.shape[0]):
        for i in range(T.shape[1]):
#            print(S[row,i])
            newS.append(translate(S[row,i], 0, leftMax, 0, 1))
        newSS.append(newS)
        newS = []
    S = np.matrix(newSS)

#    print("# non-zero values: " + str(len(np.where(S>0)[0])) + " out of " + str(T.shape[0]*T.shape[1]))    # keep

#    alpha = 0.045    # optimal alpha???? NOTE

    # Computes TP, FP, TN, FN
    TP, FP, TN, FN = 0, 0, 0, 0
    for row in range(T.shape[0]):
        for feat in range(T.shape[1]):
            # not an attack
            if S[row, feat] < alpha:
                if row in malPkts:  # an attack, False Negative
                    FN += 1
                else:   # not an attack, True Negative
                    TN += 1
            else:   # an attack
                if row in malPkts:  # an attack, True Positive
                    TP += 1
                else:   # not an attack, False Positive
                    FP += 1

    # Compute the False Positive and True Positive Rates
    FPR = FP*1./(FP + TN)
    TPR = TP*1./(TP + FN)

    print("Lambda: ", l, "  Alpha: ", alpha)
    print("\nFPR:", FPR, "    TPR:", TPR)
#    print("# of attacks: ", len(malPkts))

#    plotMat(S)     # keep
    plotS(X, s, E, maxRank)
#    plotS(X, s, E, maxRank, True)
#    attackInds = [823, 829, 886, 898, 977]
    
#    plotter(S,attackInds,xname="Stage 3")

    return TPR, FPR

# mapping values function
def translate(value, leftMin, leftMax, rightMin, rightMax):
    # Figure out how 'wide' each range is
    leftSpan = leftMax - leftMin
    rightSpan = rightMax - rightMin

    # Convert the left range into a 0-1 range (float)
    valueScaled = float(value - leftMin) / float(leftSpan)

    # Convert the 0-1 range into a value in the right range.
    return rightMin + (valueScaled * rightSpan)


# plots matrices
def plotMat(mat):
    print("Plotting...")

    plt.matshow(mat)
#    plt.imshow(mat)
#    plt.colorbar()
    plt.show()


# plots Sigma matrices from PCA (SVD) and sRPCA
def plotS(T, svd, srpca, maxRank, log=False):
    print("Plotting...")

    T = np.asmatrix(T)
#    plt.plot(range(T.shape[1]), svd, 'rs', range(maxRank), srpca, 'bo')
    plt.plot(range(T.shape[1]), svd, 'rs')
    if log:
        plt.yscale("log")
    plt.show()

    plt.plot(range(maxRank), srpca, 'bo')
    plt.show()


# plots the graphs similar to the ones in the journal
def plotJ(mat):
    print("Plotting...")

    plt.title("PCA/RPCA Title")
    plt.xlabel("Column of Data Matrix") # TODO remove values from x axis
    plt.ylabel("Value of Infinity Norm")
    plt.plot([0.5, 0.5])    # Threshold line TODO make it a red line
    # TODO figure out how to make vertical lines (make light gray and dotted)
    
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
    X, malPkts = loadFile(testing)   # TODO this is only for getting the Thurs morning file for journal

    # one-hot encode columns

    '''
        "Dist Src Port": 0,
            "Dest IP": 0,
                "Dest Port": 0,
                    "Dist Dest Port": 0,
                        "Protocol": 0,
                            "Pkt Len Mean": 0
    '''
    # Source IP
    sip = X[:,0].T
    siph = []
    # NOTE only uses the first byte of IP address
    for r in range(X.shape[0]):
        adr = sip[0,r]
        newAdr = adr.split(".")[0]
        siph.append(newAdr)
    sip = np.array(siph, dtype=float)
#    ds = dict(Counter(sip))
#    print(ds)
#    print(len(ds))
    sipOH = oneHot(sip)
    feat["Src IP"] = sipOH.shape[1]  # add to feature dict

    # Source Port
    sp = X[:,1].T
    sph = []
    for r in range(X.shape[0]):
        sph.append(sp[0,r])
    sp = np.array(sph, dtype=float)
    sp[sp > 1024] = 1024
    spOH = oneHot(sp)
    feat["Src Port"] = len(feat) + int(spOH.shape[1])  # add to feature dict

#    print(len(feat), spOH.shape[1])
#    print(feat)

    # Distinguishing source ports
    sp[sp < 1024] = 0
    sp[sp >= 1024] = 1
    dspOH = oneHot(sp)
    feat["Dist Src Port"] = dspOH.shape[1]  # add to feature dict

    # Destination IP
    dip = X[:,2].T
    diph = []
    # NOTE only uses the first byte of IP address
    for r in range(X.shape[0]):
        adr = dip[0,r]
        newAdr = adr.split(".")[0]
        diph.append(newAdr)
    dip = np.array(diph, dtype=float)
    dipOH = oneHot(dip)
    feat["Dest IP"] = dipOH.shape[1] # add to feature dict

    # Destination Port
    dp = X[:,3].T
    dph = []
    for r in range(X.shape[0]):
        dph.append(dp[0,r])
    dp = np.array(dph, dtype=float)
    dp[dp > 1024] = 1024
    dpOH = oneHot(dp)
    # Distingishing destination ports
    dp[dp < 1024] = 0
    dp[dp >= 1024] = 1
    ddpOH = oneHot(dp)

    # Protocols
    p = X[:,4].T
    ph = []
    for r in range(X.shape[0]):
        ph.append(p[0,r])
    p = np.array(ph, dtype=float)
    pOH = oneHot(p)

    # Packet Length Mean
    plm = X[:,5]
    plmh = []
    for r in range(X.shape[0]):
        plmh.append(float(plm[r,0]))
    plm = np.array(plmh, dtype=float)
    plmd = plm.shape[0]
    plm = np.reshape(plm, (plmd,1))


    # creates new X matrix
    print(sipOH.shape, spOH.shape, dipOH.shape, dpOH.shape, pOH.shape, dspOH.shape, ddpOH.shape, plm.shape)
    newX = np.concatenate((sipOH, spOH, dipOH, dpOH, pOH, dspOH, ddpOH, plm), axis=1)

    X = np.asmatrix(newX)
    X.astype(float)
    print(X.shape,X.dtype)

    # clean and normalize matrix
    X = cleanMat(X)
    X = normMat(X)

    if stats:
        printStats(testing[0])   # can only handle 1 file at a time right now


    TPR, FPR = 0, 1
#    l = 0.64
    l = 0.15
    alpha = 0.045

    runAnalysis(X, l, alpha, malPkts)

    exit(0)

    while TPR < FPR:
        TPR, FPR = runAnalysis(X, l, alpha, malPkts)

        if FPR-TPR < 0.5:
#            alpha += 0.05
            l -= 0.05
        else:
            l -= 0.05
            
