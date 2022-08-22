from data_preprocessing import simple_impute_missing_data, split_data,  normalise_data
from sklearn import datasets
import pandas as pd


data = pd.DataFrame(datasets.load_wine().data)

print('original data shape', data.shape)

# print(data.iloc[:,:-1])

if __name__ == "__main__":

    X, y = data.iloc[:, :-1], data.iloc[:, -1]
    X_train, X_test, y_train, y_test = split_data(X, y, 0.25, 7)
    print(X_train.shape)
    print(X_test.shape)
    normalized_train_X, normalized_test_X = normalise_data(X_train, X_test)
    X_train, X_test = simple_impute_missing_data(X_train, X_test, 1, 'median')

    print('Mean Imputation')
    print(X_train, X_test)

