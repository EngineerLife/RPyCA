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

num_rows = 12000
num_feat = 77

# loads a list of files and extracts/forms contents
def loadFile(names):
#    destP = []  # groups feature and counts them
#    destIP = []
    mat = []
    mx = "\n"
    for name in names:
        count = 0
    #    count = 1   # groups feature and counts them

        with open(name) as fin:
            for line in fin:
                if count == 0:
                    lineLabel = mx.join(line.split(","))
#                    lineLabel = line.split(",")     # shows label + 1 item of data


                # shows label + 1 item of data
                if count >= 1 and count <= num_rows:
                    lineData = line.split(",")

                    indexes = len(lineData)-1
                    temp = []

                    for item in range(indexes):   # uses 77 features
                        if item >= 7:   # removes first few features that aren't int/floats
                            temp.append(lineData[item])
                        if item == indexes-1:
                            mat.append(temp)
                            temp = []

#                    destP.append(lineData[1])      # source IP    # groups feature and counts them
#                    destIP.append(lineData[3])     # destination IP
                    
                    # shows label + 1 item of data
#                    c = 0
#                    for l in lineLabel:
#                        if not "Min" in l and not "Max" in l and not "Std" in l and not "Mean" in l:
#                            print(str(l)+":   "+str(lineData[c]))
#                        c += 1
#                    exit(0)
                elif count > num_rows:
                    newMat = np.matrix(mat, dtype='float')
                    print(newMat.shape)
                    break

#                count = 1
                count += 1
        break
#    print(mat)
     # groups features and counts them
#    destP = [int(x) for x in destP if not " Source IP" in x]
#    destP = [x for x in destP if not " Source IP" in x]
#    destIP = [x for x in destP if not " Destination IP" in x]
#    destP = sorted(destP)
#    a = dict(Counter(destP))
#    b = dict(Counter(destIP))
#    print("Source IP's:\n"+str(a))
#    print("# Unique Source IP's:\n"+str(len(a)))
#    print("# Unique Destination IP's:\n"+str(len(b)))
    return newMat

# prints stats on file
def printStats(filename):
    c, mal, m, a, ogMal = 0, 0, 0, 0, 0
    sourceIps, malSourceIps, malTime = [], [], []
    malLines = []

    with open(filename) as fin:
        for line in fin:
            lineInfo = line.split(",")
           
            # skips header row
            if c == 0:
                c+=1
                ogMal = mal
                continue
            
            # label
            if not "BENIGN" in lineInfo[84]:
                if not lineInfo[1] in malSourceIps:
                    malSourceIps.append(lineInfo[1])
                malLines.append(c)
                mal += 1

            # source IP
            if not lineInfo[1] in sourceIps:
                sourceIps.append(lineInfo[1])

            # timestamp
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
    print("Total # packet flows:", c)
    print("Total # malicious packet flows:", mal)
    print("Percent of malicious to total", ((mal)/(c)*100))
    print("# Unique Source IP's:", len(dictSource))
    print("# Unique Malicious Source IP's:", len(dictMalSource))
    print("Malicious packets during:", malTime)
#    print("Malicious packets on lines:", malLines)
    print("# packets during", timeFrame1, "=", m)
    print("# packets during", timeFrame2, "=", a)



# function to run PCA and RPCA and graph them
def runAnalysis(X):
    print("Running Analysis...")
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
    
    u = []
    v = []
    vecM = []
    vecEpsilon = []
    UOrig = []
    VOrig = []
#    UOrig = np.matrix(np.random.random(size=[num_rows, maxRank]))   # ideally, U would be num_rows x 523 (if have 523 features)
#    VOrig = np.matrix(np.random.random(size=[num_feat, maxRank]))   # ideally, V would be square (523 x 523)

    # fill in data from X
    UOrig = np.matrix([[X[x][y] for y in range(maxRank)] for x in range(num_rows)])
    VOrig = np.matrix([[X[x][y] for y in range(maxRank)] for x in range(num_feat)])

#    print(UOrig.shape, VOrig.shape)

    for i in range(num_rows):       # 1000
        for j in range(num_feat):   # 77
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
    print("# non-zero values: " + str(len(np.where(S>0)[0])) + " out of " + str(num_rows*num_feat))    # keep
    plotMat(S)     # keep

#    plotS(s, E, maxRank)


# plots matrices
def plotMat(mat):
    print("Plotting...")

    plt.matshow(mat)
#    plt.imshow(mat)
#    plt.colorbar()
    plt.show()


# plots Sigma matrices from PCA (SVD) and sRPCA
def plotS(svd, srpca, maxRank):
    print("Plotting...")

    plt.plot(range(num_feat), svd, 'rs', range(maxRank), srpca, 'bo')
    plt.show()

# normalizes every column in the matrix from start position to end position
def normMat(M):

    std = np.std(M,axis=0)
#    print("# <= 0 val: " + str(len(np.where(std<=0)[0])))

    # ensures no values are 0
    stdDev = []
    for i in range(num_feat):
        if std[0,i] <= 0:
            stdDev.append(1e-5)
        else:
            stdDev.append(std[0,i])
    stdDev = np.asmatrix(stdDev)

    # Z-Score
    normed = (M - np.mean(M,axis=0)) / stdDev

    # clears up matrix for exporting to csv
    nn = []
    for row in range(num_rows):
        inar = []
        for clmn in range(num_feat):
            inar.append(normed[row,clmn])
        nn.append(inar)

    with open('normed3.csv', 'w') as writeFile:
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

    # loads and formats data from file
    X = loadFile(testing)

    X = X.astype(int)

    # clean and normalize matrix
    X = cleanMat(X)
    X = normMat(X)

    if stats:
        printStats(testing[0])   # can only handle 1 file at a time right now

#    runAnalysis(X)
