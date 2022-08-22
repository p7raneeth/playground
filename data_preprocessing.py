from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.impute import SimpleImputer

def split_data(X, y, test_per, random_state):
     """
     this function is used to split the data into train and test 
     """

     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
     return X_train, X_test, y_train, y_test


def normalise_data(X_train, X_test):
     """
     this function is used to split the normalize the data
     """
     normalizer = preprocessing.Normalizer()
     normalized_train_X = normalizer.fit_transform(X_train)
     normalized_test_X = normalizer.transform(X_test)
     return normalized_train_X, normalized_test_X

def simple_impute_missing_data(X_train, X_test, col_name, strategy):
     mean_imputer = SimpleImputer(strategy=strategy)
     X_train.iloc[:,col_name] = mean_imputer.fit_transform(X_train.iloc[:,col_name].values.reshape(-1,1))
     X_test.iloc[:,col_name] = mean_imputer.fit_transform(X_test.iloc[:,col_name].values.reshape(-1,1))

     return X_train, X_test

