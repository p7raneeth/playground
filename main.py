from data_preprocessing import simple_impute_missing_data, split_data,  normalise_data, dimentionality_reduction, model_train
from sklearn import datasets
import pandas as pd
import numpy as np


# data = pd.DataFrame(datasets.load_iris().data)

data = pd.read_csv('classification.csv')

# print('original data shape', data.head())

# print(data.iloc[:,:-1])

if __name__ == "__main__":

    X, y = data.iloc[:, :-1], data.iloc[:, -1]
    X_train, X_test, y_train, y_test = split_data(X, y, 0.25, 7)
    print('--------')
    print(y_train)

    normalized_train_X, normalized_test_X = normalise_data(X_train, X_test)
    X_train, X_test = simple_impute_missing_data(X_train, X_test, 1, 'median')
    X_train, X_test, explained_variance = dimentionality_reduction(X_train, X_test, 2)
    print(np.array(explained_variance).sum())
    model_train(X_train, X_test, y_train, y_test)

    # print(X_train, X_test, explained_variance)

