# Python3 file
# Created by Marissa Bennett

import math, sys, csv, ast, time, re, warnings
import numpy as np
import pandas as pd
import rpyca as rp
from helperFiles.logger import *
from helperFiles.matrixOp import *
from helperFiles.fileHandler import *
from helperFiles.plotter import *
from helperFiles.models import *
from helperFiles.configParser import *

# main function
if __name__ == '__main__':
    start_time = time.time()
    # Ask for configuration to use
    configType, con = setConfig()
    # Set log for debugging or other purposes (can be overridden)
    setLog(con['LogFile'])
    logMsg(1,"CONFIGURATION USED: %s" % str(configType))
    # Set all other configuration variables
    fileName = con['CSVFile']
    labelsName = re.sub(r'[^\w]', '', con['Labels'])
    onehot = toList(con['OneHot'], integer=False)
#    skip = toList(con['Skip'], integer=False)
    skip = ['FlowID', 'SourceIP', 'Timestamp', 'Label']
    seed = (0 if (con['RandomSeed'] == 0) else con['RandomSeed'])
    ratioTrain, ratioValid = con['RatioTrainData'], con['RatioValidData']
    # Set ML model to run
    toRun = [con['Models']]
    if "all" == con['Models']:
        # NOTE these are not all the models in the model.py file
        toRun = ['rf','knn','logreg','svm','dtree','nb','kmeans','gb','pynn']
    # Set Looping actions 
    howToRun = []
    mode = con['Mode']
    if mode == 1:
        howToRun = [con['LambdaStartValue']]
    elif mode == 2: # this is used for plotting
        howToRun = [con['LambdaStartValue']] * 10
    else:           # default for finding a good lambda
        howToRun = frange(con['LambdaStartValue'], con['LambdaEndValue'], con['LambdaIncrValue'])

    # ensures preprocessing happens at least once
    # TODO look into if I could just randomize the random data again instead???
    pre = True

    Xlis,LSlis,XLSlis = [], [], []

    # main loop
    # TODO normalize each matrix with X1 things (see paper)
    for l in howToRun:
        if not mode == 0 or pre:
            [X1, X2, X3], ymat = preproc(fileName, labelsName, seed, ratioTrain, ratioValid, onehot, skip)
#            [X1, X2, X3], ymat = preprocKaggle(fileName, labelsName, seed, ratioTrain, ratioValid, oneHot, skip)
            pre = False     # done preprocessing for mode 0 only!

        '''
        # XXX
        import matplotlib.pyplot as plt
        # SVD PCA
        u, s, vh = np.linalg.svd(X3)
#        print("U:\n",u[:,0])
        x = np.array(u[:,0], dtype=float).flatten()
        y = np.array(u[:,1], dtype=float).flatten()

        bad_x, bad_y = [], []
        for i in range(len(ymat[2])):
            if ymat[2][i] == 1:
                bad_x.append(x[i])
                bad_y.append(y[i])
                np.delete(x, i)
                np.delete(y, i)

        plt.scatter(x, y, color='green')
        plt.scatter(bad_x, bad_y, color='red')
        plt.show()
        exit(0)
        # XXX
        '''

        logMsg(1, "Lambda: %s" % (str(l)))
        print("\n\nLAMBDA: ", l)

        # runs RPCA
        [LS1, LS2], [XLS1, XLS2] = rp.rpca(X1, X2, l)
        # XXX Future Work: see if lambda can be tuned outside of using ML models??

        # ML/AI loop
        for m in toRun:
            print("running ML")
            Xmat, LSmat, XLSmat, ymatX12 = [X1, X2], [LS1, LS2], [XLS1, XLS2], [ymat[0], ymat[1]]
            res, dall = runModels(Xmat, LSmat, XLSmat, ymatX12, code=m)
           
            # Validates ONLY if a good f1 score occurred
            if res: 
                print("Validating...")
                logMsg(1, "Validating GOOD Lambda: %s" % (str(l)))
    
                # validate
                [LS1, LS3], [XLS1, XLS3] = rp.project(X1, X3)

                # ML/AI
                Xmat, LSmat, XLSmat, ymatX13 = [X1, X3], [LS1, LS3], [XLS1, XLS3], [ymat[0], ymat[2]]
                res, dgood = runModels(Xmat, LSmat, XLSmat, ymatX13, code=m)

                Xlis.append(dgood[0])
                LSlis.append(dgood[1])
                XLSlis.append(dgood[2])

    generateResults(toRun[0].upper(),l,Xlis,LSlis,XLSlis)
    logMsg(1, "Time to complete: %s" % str(time.time() - start_time))
