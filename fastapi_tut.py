from fastapi import FastAPI
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
# from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

from sklearn.model_selection import RandomizedSearchCV
# app = FastAPI()
app = FastAPI()



# async def main(hyper_parm1, hyper_parm2):
#     print(hyper_parm1, hyper_parm2)
#     return {f"accuracy {hyper_parm1} b is {hyper_parm2}": "100%"}

@app.get("/confusion_matrix")
def evaluate(model, test_features, test_labels):
    predictions = model.predict(test_features)
    print(predictions)
    # errors = abs(predictions - test_labels)
    # mape = 100 * np.mean(errors / test_labels)
    # accuracy = 100 - mape
    # print('Model Performance')
    # print('Average Error: {:0.4f} degrees.'.format(np.mean(errors)))
    # print('Accuracy = {:0.2f}%.'.format(accuracy))
    
    return predictions


@app.get("/hyprer_parms")
def train_dtcl(n_est):
    df = pd.read_csv("heart_disease.csv")
    X = df.drop(columns="target")
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, 
                                                        shuffle=True, random_state=0)

    # Create the random grid
    random_grid = {'n_estimators': [int(x) for x in n_est],
               }


    rf = RandomForestClassifier()
    # Random search of parameters, using 3 fold cross validation, 
    # search across 100 different combinations, and use all available cores
    
    rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid)
    # Fit the random search model
    rf_random.fit(X_train, y_train)
    best_random = rf_random.best_estimator_
    print('best - random: ----- ', best_random)
    y_pred = evaluate(best_random, X_test, y_test)


    cf_matrix = confusion_matrix(y_test, y_pred)
    print("\nTest confusion_matrix")
    sns.heatmap(cf_matrix, annot=True, cmap='Blues')
    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('True', fontsize=12)



