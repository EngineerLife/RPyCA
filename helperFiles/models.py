import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
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
    # elbow method
#    Sum_sqrd_dist = []
#    K = range(1,15)
#    for k in K:
#        km = KMeans(n_clusters=k)
#        km = km.fit(X)
#        Sum_sqrd_dist.append(km.inertia_)
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




####
# Create Y labels
####
# creates labels for data set
#   Input: list of attak point indexes
#           length of data
#   Output: list of labels [0, 1] 0 being benign and 1 being malicious
def createY(lenData, atkPnts):
    y = []
    j = 0
    for i in range(lenData):
        if j < len(atkPnts) and i == atkPnts[j]:     # NOTE we can do this bc atkPnts are in numerical order
            y.append(1)
#            y.append("attack")
            j += 1
        else:
            y.append(0)
#            y.append("normal")
    return y
#    return np.flip(y)

# runs all the models or chosen models
#   Input: X mat, 
#          L mat, 
#          S mat, 
#          malicious packets counts, (see confPaper.py)
#          integer to split data on for train and test <=== FIXME
#          code defaults to empty (all models run) or contains codes for specific ones to run
#          tune if we are tuning the model with the data to find optimal values
#   Output: Prints confusion matricies for each model and f1_scores
#def runModels(X, L, S, mpc, splitOn, code=[], tune=False):
def runModels(X, L, S, mpc, splitOn, code=[], tune=False):
    # creates training and testing label data to be used for all models
    if not type(mpc) == str: # TODO check if list or if filename
        y = createY(len(X), mpc)
    else:
        y = loadLabels(mpc)
    # TODO CHECK THAT THESE ARE CORRECT
    y_train, y_test = np.split(y, [splitOn[0]])
    y_test, y_validate = np.split(y_test, [splitOn[1]])
    print(y_train.shape, y_test.shape, y_validate.shape)

    LS1 = np.concatenate((L[0],S[0]), axis=1)
    XLS1 = np.concatenate((X[0], L[0], S[0]), axis=1)
    LS2 = np.concatenate((L[1],S[1]), axis=1)
    XLS2 = np.concatenate((X[1], L[1], S[1]), axis=1)
    train = [X[0], LS1, XLS1]     # holds data matricies to run models on
    test = [X[1], LS2, XLS2]
#    validate = [X3, LS3, XLS3]
    matName = ["X", "CONCAT LS", "CONCAT XLS"]
    # can choose code(s) to use
    if not code:
        code = np.arange(8)    # NOTE if there are more models increase this number
   
    for i in code:
        countName = 0
        for matType in range(3):
            X_train = train[matType]
            X_test = test[matType]
#            X_train, X_test = np.split(dataMat, [splitOn[0]])
#            X_test, X_validate = np.split(X_test, [splitOn[1]])
#            print("SHAPES (train, test validate): X[", X_train.shape, X_test.shape, X_validate.shape, "]\n \
#            y[", y_train.shape, y_test.shape, y_validate.shape, "]")

            print("Model SHAPES:", X_train.shape, X_test.shape, y_train.shape, y_test.shape)

            # TODO somehow get validate stuff in here???
            y_pred, regr = chooseModel(str(i), X_train, X_test, y_train, tune)
            # ONLY for tuning; not for training/main algo
            if tune and not regr == None:
                print("Oops! Did not want to test this right now.")
                exit(0)
                findOptVal(regr, X_train, y_train)

            print("FOR MATRIX: ", matName[countName])
            logMsg(1, "FOR MATRIX: %s" % (matName[countName]))
            #print("MEAN ACCURACY: ", classifier.score(X_test, y_test))

            # Create confusion matrix
            cfm = pd.crosstab(y_test, y_pred, rownames=['Actual Packet Type'], colnames=['Predicted Packet Type'])
            logMsg(1, "\n%s" % str(cfm))
            print(cfm)
            f1 = f1_score(y_test, y_pred)
            logMsg(1, "f1_Score: %f" % f1)
            print("f1_Score: ", f1)
            countName += 1
####
# chooses the model to run 
#   [and any other qualities of it (used for tuning only)]
####
def chooseModel(code, X_train, X_test, y_train, tune=False):
    if code == "rf" or code == "0":
        if tune: return None, RandomForestRegressor()
        print("************ RANDOM FOREST ************")
        y_pred = rf(X_train, X_test, y_train)
    elif code == "knn" or code == "1":
        if tune: return None, RandomForestRegressor()
        print("************ KNN ************")
        y_pred = runKNN(X_train, X_test, y_train, 5)     # TODO accomidate for different number clusters
    elif code == "svm" or code == "2":
        if tune: return None, RandomForestRegressor()
        print("************ SVM ************")
        y_pred = runSVM(X_train, X_test, y_train)
    elif code == "logreg" or code == "3":
        if tune: return None, RandomForestRegressor()
        print("************ LOG REG ************")
        y_pred = runLogReg(X_train, X_test, y_train)
    elif code == "dtree" or code == "4":
        if tune: return None, RandomForestRegressor()
        print("************ DEC. TREE ************")
        y_pred = runDTree(X_train, X_test, y_train)
    elif code == "nb" or code == "5":
        if tune: return None, RandomForestRegressor()
        print("************ NAIVE BAYES ************")
        y_pred = runNB(X_train, X_test, y_train)
    elif code == "kmeans" or code == "6":
        if tune: return None, RandomForestRegressor()
        print("************ KMEANS ************")
        y_pred = runKmeans(X_train, X_test, y_train)
    elif code == "gb" or code == "7":
        if tune: return None, RandomForestRegressor()
        print("************ GRAD. BOOSTING ************")
        y_pred = gb(X_train, X_test, y_train)
    elif code == "nn" or code == "8":
        if tune: return None, RandomForestRegressor()
        print("************ NN ************")
        y_pred = runNN(X_train, X_test, y_train)
    else:   # SHOULD NEVER GET HERE
        print("************ ELSE ************")
        exit(0)
    return y_pred, None



    # neural network ensamble
