import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from .fileHandler import *
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
#          integer to split data on for train and test
#          code defaults to empty (all models run) or contains codes for specific ones to run
#   Output: Prints confusion matricies for each model and f1_scores
def runModels(X, L, S, mpc, splitOn, code=[]):#X_train, X_test, y_train, y_test):
    # creates training and testing label data to be used for all models
    if not type(mpc) == str: # TODO check if list or if filename
        y = createY(len(X), mpc)
    else:
        y = loadLabels(mpc)
    y_train, y_test = np.split(y, [splitOn])

    LS = np.concatenate((L,S), axis=1)
    XLS = np.concatenate((X, L, S), axis=1)
    matAr = [X, LS, XLS]    # holds data matricies to run models on
    matName = ["X", "CONCAT LS", "CONCAT XLS"]
    # can choose code(s) to use
    if code:    # TODO change this to one int later
        print(code)
    else:
        code = np.arange(8)    # NOTE if there are more models increase this number
   
    for i in code:
        countName = 0
        for dataMat in matAr:
            X_train, X_test = np.split(dataMat, [splitOn])
            print("SHAPES: ", X_train.shape, X_test.shape, y_train.shape, y_test.shape)

            y_pred = chooseModel(str(i), X_train, X_test, y_train)
            print("FOR MATRIX: ", matName[countName])

            #print("MEAN ACCURACY: ", classifier.score(X_test, y_test))
            # Create confusion matrix
            cfm = pd.crosstab(y_test, y_pred, rownames=['Actual Packet Type'], colnames=['Predicted Packet Type'])
            print(cfm)
            f1 = f1_score(y_test, y_pred)
            print("f1_Score: ", f1)
            countName += 1


def chooseModel(code, X_train, X_test, y_train):
    if code == "rf" or code == "0":
        y_pred = rf(X_train, X_test, y_train)
        print("************ RANDOM FOREST ************")
    elif code == "knn" or code == "1":
        print("************ KNN ************")
        y_pred = runKNN(X_train, X_test, y_train, 5)     # TODO accomidate for different number clusters
    elif code == "svm" or code == "2":
        print("************ SVM ************")
        y_pred = runSVM(X_train, X_test, y_train)
    elif code == "logreg" or code == "3":
        print("************ LOG REG ************")
        y_pred = runLogReg(X_train, X_test, y_train)
    elif code == "dtree" or code == "4":
        print("************ DEC. TREE ************")
        y_pred = runDTree(X_train, X_test, y_train)
    elif code == "nb" or code == "5":
        print("************ NAIVE BAYES ************")
        y_pred = runNB(X_train, X_test, y_train)
    elif code == "kmeans" or code == "6":
        print("************ KMEANS ************")
        y_pred = runKmeans(X_train, X_test, y_train)
    elif code == "gb" or code == "7":
        print("************ GRAD. BOOSTING ************")
        y_pred = gb(X_train, X_test, y_train)
    elif code == "nn" or code == "8":
        print("************ NN ************")
        y_pred = runNN(X_train, X_test, y_train)
    else:
        print("************ ELSE ************")
        exit(0)
    return y_pred



    # neural network ensamble
