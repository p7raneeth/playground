from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score


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
     """
     this function is used to impute the missing data 
     """
     mean_imputer = SimpleImputer(strategy=strategy)
     X_train.iloc[:,col_name] = mean_imputer.fit_transform(X_train.iloc[:,col_name].values.reshape(-1,1))
     X_test.iloc[:,col_name] = mean_imputer.fit_transform(X_test.iloc[:,col_name].values.reshape(-1,1))

     return X_train, X_test


def dimentionality_reduction(X_train, X_test, n_comp):
     """
     this function helps in reducing the dimentions to a desired n_components value
     """
     pca = PCA(n_comp)    
     X_train = pca.fit_transform(X_train)
     X_test = pca.transform(X_test)
     explained_variance = pca.explained_variance_ratio_

     # print(X_train, X_test, explained_variance)
     return(X_train, X_test, explained_variance)


def model_train(X_train, X_test, y_train, y_test):
     classifier = RandomForestClassifier(max_depth=2, random_state=0)
     classifier.fit(X_train, y_train)

     # Predicting the Test set results
     y_pred = classifier.predict(X_test)
     print(len(y_pred))

     cm = confusion_matrix(y_test, y_pred)
     print(cm)
     print('Accuracy' , accuracy_score(y_test, y_pred))


