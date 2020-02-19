# Python3 file
# Created by Marissa Bennett
# PACKAGE METHODS TO BE USED IN HERE
# (names of these can be changed later)
#       setConfiguration    - for users to set it
#       updateConfiguration - for users to update it
#       deleteConfiguration - for users to get rid of some
# (run RPCA Projection Models [TODO need better package name...])
#       runINIT             - first run of data set
#       runRP               - for users to do the thing with their own models
#       runRPM              - for users to do the thing
#
#       printPlots

# TODO need init function

import math, sys, csv, ast, re, warnings
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.decomposition import PCA
from helperFiles.sRPCAviaADMMFast import *
from helperFiles.RPCA import *
from helperFiles.logger import *
from helperFiles.matrixOp import *
from helperFiles.fileHandler import *
from helperFiles.plotter import *
from helperFiles.models import *
from helperFiles.configParser import *

import pandas as pd     # XXX used for plotting only


# !!!!!!TODO make the input for creating X the same (csv or something)


# main function
if __name__ == '__main__':
    # Ask for configuration to use
    configType, con = setConfig()
    # Set log for debugging or other purposes (can be overridden)
    setLog(con['LogFile'])
    logMsg(1,"CONFIGURATION USED: %s" % str(configType))
    # Set all other configuration variables
    mode = con['Mode']
    fileName = con['CSVFile']
    header, labelsLoc, rowClmn = int(con['Header']), int(con['Labels']), int(con['RowClmn'])
    onehot = toList(con['OneHot'])
    skip = toList(con['Skip'])
    seed = (0 if (con['RandomSeed'] == 0) else con['RandomSeed'])
    ratioTrain, ratioTest = con['RatioTrainData'], con['RatioTestData']
    # ML model to run
    toRun = [con['Models']]
    if "all" == con['Models']:
        toRun = ['rf','knn','svm','logreg','svm','dtree','nb','kmeans','gb']
    
    howToRun = []
    if mode == 1:
        howToRun = [con['LambdaStartValue']]
    elif mode == 2: # this is used for plotting
        print("SUP")
        howToRun = [con['LambdaStartValue']] * 10
    else:           # default for finding a good lambda
        howToRun = frange(con['LambdaStartValue'], con['LambdaEndValue'], con['LambdaIncrValue'])

    # ensures preprocessing happens at least once
    # TODO look into if I could just randomize the random data again instead???
    pre = True

    # main loop
    # TODO normalize each matrix with X1 things (see paper)
    for l in howToRun:
        if not mode == 0 or pre:
            [X1, X2, X3], ymat = preproc(fileName, header, labelsLoc, rowClmn, seed, ratioTrain, ratioTest, onehot, skip)
            pre = False     # done preprocessing for mode 0 only!

        logMsg(1, "Lambda: %s" % (str(l)))
        print("\n\nLAMBDA: ", l)

        # runs RPCA
        S1, L1, VThat = runAnalysis(X1, l)
        logMsg(0, "X1 SHAPES: X: %s  L: %s  S: %s" % (str(X1.shape), str(L1.shape), str(S1.shape)))

        # CHECK for S1 and L1
#        X1VTT = np.dot(X1, VThat.T)
#        LC1 = np.dot(X1VTT, VThat)
#        SC1 = X1 - LC1
#        print("L:\n%s\n%s" % (str(L1), str(LC1)))
#        print("S:\n%s\n%s" % (str(S1), str(SC1)))

        # Test Matrix;   Equation: L2 = X2*(V)^hat*(VT)^hat
        X2VTT = np.dot(X2, VThat.T)
        L2 = np.dot(X2VTT, VThat)
        S2 = X2 - L2
        logMsg(0, "X2 SHAPES: X: %s  L: %s  S: %s" % (str(X2.shape), str(L2.shape), str(S2.shape)))

        for m in toRun:
            # ML/AI
            Xmat, Lmat, Smat, ymatX12 = [X1, X2], [L1, L2], [S1, S2], [ymat[0], ymat[1]]
            res, dall = runModels(Xmat, Lmat, Smat, ymatX12, code=m)

            if res:     # Validates ONLY if a good f1 score occurred 
                print("Validating...")
                logMsg(1, "Validating GOOD Lambda: %s" % (str(l)))
    
                # validate
                X3VTT = np.dot(X3, VThat.T)
                L3 = np.dot(X3VTT, VThat)
                S3 = X3 - L3
                logMsg(0, "X3 SHAPES: X: %s  L: %s  S: %s" % (str(X3.shape), str(L3.shape), str(S3.shape)))

                # ML/AI
                Xmat, Lmat, Smat, ymatX13 = [X1, X3], [L1, L3], [S1, S3], [ymat[0], ymat[2]]
                res, dgood = runModels(Xmat, Lmat, Smat, ymatX13, code=m)
    exit(0)


    # PLOT for training lambda
    '''
                
                    fig = plt.figure()
#                    fig.subplots_adjust(left=0.07, bottom=0.21, right= 0.95, top=0.83, wspace=0.36, hspace=0.2)
                    testHist = fig.add_subplot(1, 2, 1)
                    validHist = fig.add_subplot(1, 2, 2)
                    plotHist(dall, "Testing Set", testHist)
                    plotHist(dgood, "Validation Set", validHist)
                    name = "runFig" + str(j) + ".png"
                    plt.savefig(name)
#                    plt.show()

                    plt.hist(dall[0], label='x')
                    plt.hist(histdls, label='concat ls')
                    plt.hist(histdxls, label='concat xls')
                    plt.legend(loc='upper right')
                    plt.xlabel('F1 Score')
#                    plt.ylabel('# of Runs')
                    plt.title('Testing Set')
                    name = "testSet" + str(j) + ".png"
                    plt.savefig(name)
#                    plt.show()

                    plt.hist(gooddx, label='x')
                    plt.hist(gooddls, label='concat ls')
                    plt.hist(gooddxls, label='concat xls')
                    plt.legend(loc='upper right')
                    plt.xlabel('F1 Score')
#                    plt.ylabel('# of Runs')
                    plt.title('Validation Set')
                    name = "validateSet" + str(j) + ".png"
                    plt.savefig(name)
#                    plt.show()
                    '''
