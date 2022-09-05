from fastapi import FastAPI
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
# from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import RandomizedSearchCV
app = FastAPI()

@app.get("/hyprer_parms")
def train_dtcl(n_est, max_depth, min_samples_leaf, min_samples_split):
    df = pd.read_csv("heart_disease.csv")
    X = df.drop(columns="target")
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, 
                                                        shuffle=True, random_state=0)

    # Create the random grid
    random_grid = {'n_estimators': n_est,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': True}


    rf = RandomForestRegressor()

    from sklearn.model_selection import RandomizedSearchCV
    # Random search of parameters, using 3 fold cross validation, 
    # search across 100 different combinations, and use all available cores
    rf_random = RandomizedSearchCV(estimator = [1,2], param_distributions = {'min_leaf_split': [1,2]})
    rf_random.fit(X_train, y_train)
    best_random = rf_random.best_estimator_
    random_accuracy = evaluate(best_random, X_test, y_test)

    # dtclf = DecisionTreeClassifier(random_state=42)
    # dtclf.fit(X_train, y_train)
    # y_pred = dtclf.predict(X_test) # Predictions
    # y_true = y_test # True values


    # print("Train accuracy:", np.round(accuracy_score(y_train, 
    #                                                 dtclf.predict(X_train)), 2))
    # print("Test accuracy:", np.round(accuracy_score(y_true, y_pred), 2))


    # cf_matrix = confusion_matrix(y_true, y_pred)
    # print("\nTest confusion_matrix")
    # sns.heatmap(cf_matrix, annot=True, cmap='Blues')
    # plt.xlabel('Predicted', fontsize=12)
    # plt.ylabel('True', fontsize=12)

    # rs = RandomizedSearchCV(dtclf, param_distributions=hyperparameter_space,
    #                         n_iter=10, scoring="accuracy", random_state=0,
    #                         n_jobs=-1, cv=10, return_train_score=True)

    # rs.fit(X_train, y_train)
    # print("Optimal hyperparameter combination:", rs.best_params_)
    # print()
    # print("Mean cross-validated training accuracy score:",
    #     rs.best_score_)
    # rs.best_estimator_.fit(X_train, y_train)
    # y_pred = rs.best_estimator_.predict(X_test) # Predictions
    # y_true = y_test # True values