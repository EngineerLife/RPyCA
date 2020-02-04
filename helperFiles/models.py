import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re
from .fileHandler import *
from .logger import *
from scipy.spatial.distance import cdist
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier
from sklearn.cluster import KMeans
from sklearn import svm
#from sklearn import metrics

# Each of these will have their own lambdas

####
# NN
####
def runNN(X_train, X_test, y_train):
    # use pytorch
    print("please no")

####
# Logistic Regression
####
def runLogReg(X_train, X_test, y_train):
    clf = LogisticRegression(random_state=0, solver='lbfgs',
                             multi_class='multinomial')
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    return y_pred

####
# Decision Tree
####
def runDTree(X_train, X_test, y_train):
    clf = DecisionTreeClassifier(random_state=0)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    return y_pred

####
# Naive Bayes
####
def runNB(X_train, X_test, y_train):
    gnb = GaussianNB()
    gnb.fit(X_train, y_train)
    y_pred = gnb.predict(X_test)
    return y_pred

####
# SVM
####
def runSVM(X_train, X_test, y_train):
    clf = svm.SVC(gamma="scale")
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    return y_pred

####
# K Means
####
# k-means 
def runKmeans(X_train, X_test, y_train):
    clf = KMeans(n_clusters=2, random_state=0)
    clf.fit(X_train)
    y_pred = clf.predict(X_test)
    return y_pred

####
# kNN
####
# X_train is a m by n matrix of training data
# X_test is a m by n matrix of testing data
# y_train is a m by 1 vector of training data labels
# y_test is a m by 1 vector of testing data labels
#
def runKNN(X_train, X_test, y_train, cluster):
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    clf = KNeighborsClassifier(n_neighbors=cluster, leaf_size=1)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    return y_pred

####
# random forest
####
# Takes in data to fit and params
# performs random forest classification
def rf(X_train, X_test, y_train):#, maxDepth, nEst, randState):
    clf = RandomForestClassifier(n_estimators=100, max_depth=2, min_samples_leaf=5, min_samples_split=10)#, random_state=0)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    return y_pred

####
# Gradient Boosting????
####
def gb(X_train, X_test, y_train):
    clf = GradientBoostingClassifier()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    return y_pred

####
# auto-encoder (MLP)
###


####
# Find Optimal Values TODO TODO TODO
####
# checks all combinations of model on data to find optimal tuning values
#   Input: regr is the regressor model object
#          X_train data
#          y_train labels to the X_train data
#   Output: None
def findOptVal(regr, X_train, y_train):
    # Create the parameter grid based on the results of random search 
    param_grid = {
        'bootstrap': [True],
        'max_depth': [2, 10, 20],#, 50],
    #   'max_features': [2, 3],
        'min_samples_leaf': [3, 4, 5],
        'min_samples_split': [8, 10, 12],
        'n_estimators': [100, 200, 300, 1000]
    }
                                                                   
    # Create a based model
#    rf = RandomForestRegressor()

    # Instantiate the grid search model
    grid_search = GridSearchCV(estimator=regr, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)

    # Fit the grid search to the data
    grid_search.fit(X_train, y_train)
    print(grid_search.best_params_)

    best_grid = grid_search.best_estimator_
    print(grid_accuracy)


# runs all the models or chosen models
#   Input: X mat = 2 mats, 
#          L mat = 2 mats, 
#          S mat = 2 mats, 
#          y mat = 2 vectors,
#          code defaults to empty (all models run) or contains codes for specific ones to run
#          tune if we are tuning the model with the data to find optimal values
#   Output: Prints confusion matricies for each model and f1_scores
def runModels(X, L, S, ymats, code='', tune=False):
    y_train, y_test = ymats[0], ymats[1]
#    print(y_train.shape, y_test.shape)

    LStrain = np.concatenate((L[0],S[0]), axis=1)
    XLStrain = np.concatenate((X[0], L[0], S[0]), axis=1)
    LStest = np.concatenate((L[1],S[1]), axis=1)
    XLStest = np.concatenate((X[1], L[1], S[1]), axis=1)
    train = [X[0], LStrain, XLStrain]     # holds data matricies to run models on
    test = [X[1], LStest, XLStest]

    matName = ["X", "CONCAT LS", "CONCAT XLS"]
    if not code:
        # for running a custom model. Saves off training and testing
        print("RUN CUSTOM MODEL")
    # runs all models if no code given
    # FIXME does not work with new design in main.py
#    elif code[0] == "all":
#        code = np.arange(9)    # NOTE if there are more models increase this number
    ifgood = False

    # XXX for plotting
    d = []
    print(code)
    countName = 0
    for matType in range(3):
        X_train = train[matType]
        X_test = test[matType]
#            print("Model SHAPES:", X_train.shape, X_test.shape, y_train.shape, y_test.shape)
        y_pred, regr, m = chooseModel(code, X_train, X_test, y_train, tune)
        logMsg(1, m)    # logs the chosen model
        # ONLY for tuning; not for training/main algo
#        if tune and not regr == None:
#            print("Oops! Did not want to test this right now.")
#            exit(0)
#            findOptVal(regr, X_train, y_train)

#            print("FOR MATRIX: ", matName[countName])
        logMsg(1, "FOR MATRIX: %s" % (matName[countName]))

        # Create confusion matrix
        cfm = pd.crosstab(y_test, y_pred, rownames=['Actual Packet Type'], colnames=['Predicted Packet Type'])
        logMsg(1, "\n%s" % str(cfm))
#            print(cfm)
        f1 = f1_score(y_test, y_pred)
        logMsg(1, "f1_Score: %f" % f1)

        # XXX creates row for plotting
        d.append(f1)

        if float(f1) > 0.0:
            print("f1_Score for %s : %s" % (matName[countName], str(f1)))
#            if float(f1) >= 0.51:
            ifgood = True
#                print("GOOD f1_Score for %s : %s" % (matName[countName], str(f1)))
        countName += 1

    return ifgood, d

# TODO FIXME clean this function up later
####
# chooses the model to run 
#   [and any other qualities of it (used for tuning only)]
####
def chooseModel(code, X_train, X_test, y_train, tune=False):
    m = ""
    if code == "rf" or code == "0":
        if tune: return None, RandomForestRegressor()
        m = "************ RANDOM FOREST ************"
        y_pred = rf(X_train, X_test, y_train)
    elif code == "knn" or code == "1":
        if tune: return None, RandomForestRegressor()
        m = "************ KNN ************"
        y_pred = runKNN(X_train, X_test, y_train, 5)     # TODO accomidate for different number clusters
    elif code == "svm" or code == "2":
        if tune: return None, RandomForestRegressor()
        m = "************ SVM ************"
        y_pred = runSVM(X_train, X_test, y_train)
    elif code == "logreg" or code == "3":
        if tune: return None, RandomForestRegressor()
        m = "************ LOG REG ************"
        y_pred = runLogReg(X_train, X_test, y_train)
    elif code == "dtree" or code == "4":
        if tune: return None, RandomForestRegressor()
        m = "************ DEC. TREE ************"
        y_pred = runDTree(X_train, X_test, y_train)
    elif code == "nb" or code == "5":
        if tune: return None, RandomForestRegressor()
        m = "************ NAIVE BAYES ************"
        y_pred = runNB(X_train, X_test, y_train)
    elif code == "kmeans" or code == "6":
        if tune: return None, RandomForestRegressor()
        m = "************ KMEANS ************"
        y_pred = runKmeans(X_train, X_test, y_train)
    elif code == "gb" or code == "7":
        if tune: return None, RandomForestRegressor()
        m = "************ GRAD. BOOSTING ************"
        y_pred = gb(X_train, X_test, y_train)
    elif code == "nn" or code == "8":
        if tune: return None, RandomForestRegressor()
        m = "************ NN ************"
        y_pred = runNN(X_train, X_test, y_train)
    else:   # SHOULD NEVER GET HERE
        m = "************ ELSE ************"
        exit(0)
    return y_pred, None, m



    # neural network ensamble
