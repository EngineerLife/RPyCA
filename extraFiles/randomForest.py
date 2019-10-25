# Random Forest code
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.datasets import make_classification
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GridSearchCV


# LS CONCAT: {'n_estimators': 100, 'max_depth': 2, 'min_samples_leaf': 5, 'bootstrap': True, 'min_samples_split': 10}
# LS ADD: {'min_samples_leaf': 5, 'bootstrap': True, 'min_samples_split': 10, 'max_depth': 2, 'n_estimators': 100}


# Takes in data to fit and params
# performs random forest classification
def rf(X_train, X_test, y_train, y_test):#, maxDepth, nEst, randState):
#    X, y = make_classification(n_samples=1000, n_features=4, n_informative=2, n_redundant=0, random_state=0, shuffle=False)

    clf = RandomForestClassifier(n_estimators=100, max_depth=2, min_samples_leaf=5, min_samples_split=10)#, random_state=0)
    
    clf.fit(X_train, y_train)  

#    print(clf.feature_importances_)
#    print(clf.predict(X_test))
#    print("*********************************")
    print("MEAN ACCURACY: ", clf.score(X_test, y_test))
    y_pred = clf.predict(X_test)

    # Create confusion matrix
    cfm = pd.crosstab(y_test, y_pred, rownames=['Actual Packet Type'], colnames=['Predicted Packet Type'])
    print(cfm)
    f1 = f1_score(y_test, y_pred)
    print("f1_Score: ", f1)

    return f1

#    print(confusion_matrix(y_test, y_pred))
#    print(classification_report(y_test, y_pred))



    '''
    # Create the parameter grid based on the results of random search 
    param_grid = {
            'bootstrap': [True],
            'max_depth': [2, 10, 20],#, 50],
#            'max_features': [2, 3],
            'min_samples_leaf': [3, 4, 5],
            'min_samples_split': [8, 10, 12],
            'n_estimators': [100, 200, 300, 1000]
    }
    
    # Create a based model
    rf = RandomForestRegressor()
    # Instantiate the grid search model
    grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, cv = 3, n_jobs = -1, verbose = 2)

    # Fit the grid search to the data
    grid_search.fit(X_train, y_train)

    print(grid_search.best_params_)

    best_grid = grid_search.best_estimator_

    print(grid_accuracy)
    '''


#if __name__ == "__main__":
#    rf()
