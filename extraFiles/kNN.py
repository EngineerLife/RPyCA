# K Nearest Neighbors code
import numpy as np  
import matplotlib.pyplot as plt  
import pandas as pd 
from sklearn.model_selection import train_test_split  
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GridSearchCV

# X_train is a m by n matrix of training data
# X_test is a m by n matrix of testing data
# y_train is a m by 1 vector of training data labels
# y_test is a m by 1 vector of testing data labels
#
def runKNN(X_train, X_test, y_train, y_test, cluster):
#def runKNN(X, y):
#    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

    scaler = StandardScaler()
    scaler.fit(X_train)

    X_train = scaler.transform(X_train)  
    X_test = scaler.transform(X_test)  
    
    classifier = KNeighborsClassifier(n_neighbors=cluster, leaf_size=1)
    classifier.fit(X_train, y_train)

    y_pred = classifier.predict(X_test)

    print("MEAN ACCURACY: ", classifier.score(X_test, y_test))
    # Create confusion matrix
    cfm = pd.crosstab(y_test, y_pred, rownames=['Actual Packet Type'], colnames=['Predicted Packet Type'])
    print(cfm)
    
    print("f1_Score: ", f1_score(y_test, y_pred))
    
#    print(confusion_matrix(y_test, y_pred))  
#    print(classification_report(y_test, y_pred))

    '''
    error = []

    # Calculating error for K values between 1 and 40
    for i in range(1, 40):
        knn = KNeighborsClassifier(n_neighbors=i)
        knn.fit(X_train, y_train)
        pred_i = knn.predict(X_test)
        error.append(np.mean(pred_i != y_test))

    plt.figure(figsize=(12, 6))
    plt.plot(range(1, 40), error, color='red', linestyle='dashed', marker='o',
             markerfacecolor='blue', markersize=10)
    plt.title('Error Rate K Value')
    plt.xlabel('K Value')
    plt.ylabel('Mean Error')
    
    plt.show()

    # Create the parameter grid based on the results of random search 
    param_grid = {
            'n_neighbors': [cluster],
            'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
            'leaf_size': [1, 5, 10, 15, 20, 25, 30]
    }
                                                                            
    # Create a based model
    knn = KNeighborsRegressor()
    # Instantiate the grid search model
    grid_search = GridSearchCV(estimator = knn, param_grid = param_grid, cv = 3, n_jobs = -1, verbose = 2)
    # Fit the grid search to the data
    grid_search.fit(X_train, y_train)
    print(grid_search.best_params_)
    best_grid = grid_search.best_estimator_
    '''

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

if __name__ == "__main__":
    print("Running KNN...")
#    X = [[0], [1], [2], [3]]
#    y = [0, 0, 1, 1]
#    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
#    runKNN(X_train, X_test, y_train, y_test)
