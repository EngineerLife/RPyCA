#!/bin/python3

"""
RPyCA Main application

This application runs RPCA as a pre-procesing technique before various ML techniques
with the goal of improving network security traffic anomaly detection.

To run: python3 Main.py <config name>
<config name> must correspond to config.ini config to run

Original Author: Marissa Bennett
Updated by: EngineerLife
"""

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
import cProfile

def setup():
    # Ask for configuration to use
    configType, con = setConfig()
    # Set log for debugging or other purposes (can be overridden)
    setLog(con['LogFile'])
    logMsg(1,"CONFIGURATION USED: %s" % str(configType))
    # Set all other configuration variables
    fileName = con['CSVFile']
    logMsg(1,"File Name: %s" % str(fileName))
    labelsName = re.sub(r'[^\w]', '', con['Labels'])
    ##### 
    ## TODO need to make these better and more reliable
#    onehot = toList(con['OneHot'], integer=False)
#    skip = toList(con['Skip'], integer=False)
    ######
    seed = (0 if (con['RandomSeed'] == 0) else con['RandomSeed'])
    sample = (0 if (con['SampleSize'] == 0) else con['SampleSize'])
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
    return howToRun, mode, fileName, labelsName, sample, ratioTrain, ratioValid, toRun, pre, seed

def loadData(mode, pre, fileName,labelsName,sample,seed,ratioTrain,ratioValid):
    if not mode == 0 or pre:
        if "ISCX" in fileName:  # TODO these should be changed into a function or something in the future
            skip = ['FlowID', 'SourceIP', 'Timestamp', 'Label']
            [X1, X2, X3], ymat = preproc(fileName, labelsName, sample, seed, ratioTrain, ratioValid, skip=skip)# onehot, skip)
        elif "LLS_DDOS" in fileName:
            skip = ['No.', 'Label']
            [X1, X2, X3], ymat = preprocLLSDOS(fileName, labelsName, sample, seed, ratioTrain, ratioValid, skip=skip)# onehot, skip)
        pre = False     # done preprocessing for mode 0 only!
    return [X1, X2, X3], ymat

if __name__ == '__main__':
    ######################### This section is setup #########################
    start_time = time.time()
    howToRun, mode, fileName, labelsName, sample, ratioTrain, ratioValid, toRun, pre, seed = setup()
    Xlis,LSlis,XLSlis = [], [], []
    ###############################################################################
    # load data once
    [X1, X2, X3], ymat = loadData(mode, pre, fileName,labelsName,sample,seed,ratioTrain,ratioValid)

    Xmat_train = [X1, X2]
    Xmat_validate = [X1, X3]
    ymatX12 = [ymat[0], ymat[1]]
    ymatX13 = [ymat[0], ymat[2]]
    #################### main processing loop starts here #########################
    # looping through lambda values
    for l in howToRun:
        logMsg(1, "Lambda: %s" % (str(l)))
        print("\n\nLAMBDA: ", l)
        # runs RPCA as pre-processing technique
        [LS1, LS2], [XLS1, XLS2] = rp.rpca(X1, X2, l)
        
        # calulate X (orig data), L+S (RPCA decomp), X+L+S (orig + decomp)
        # this step uses projections instead of RPCA for performance
        # depends on eigenvectors of low rank matrix from rpca call
        # leave this right below rp.rpca or this will not work
        [LS1, LS3], [XLS1, XLS3] = rp.project(X1, X3)

        LSmat_train = [LS1, LS2]
        LSmat_validate = [LS1, LS3]
        XLSmat = [XLS1, XLS2]
        XLSmat_validate = [XLS1, XLS3]

        # ML/AI loop
        for m in toRun:
            print("running ML")

            # training phase
            res, dall = runModels(Xmat_train, LSmat_train, XLSmat, ymatX12, code=m)
           
            # Validates ONLY if a good f1 score occurred
            if res: 
                print("Validating...")
                logMsg(1, "Validating GOOD Lambda: %s" % (str(l)))

                # run matrices though ML model
                _, dgood = runModels(Xmat_validate, LSmat_validate, XLSmat_validate, ymatX13, code=m)

                # store result of ML model validation run
                Xlis.append(dgood[0])
                LSlis.append(dgood[1])
                XLSlis.append(dgood[2])

###############################################################
########################### Capture results in output ##################
    generateResults(toRun[0],l,Xlis,LSlis,XLSlis)
    logMsg(1, "Time to complete: %s" % str(time.time() - start_time))

