import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def train_test_split(df, test_size=0.2, seed=42):
    """
    Returns a shuffled dataset split into train and test sets
    Inputs:
        - df (pd.dataframe): contains the dataset
        - test_size (float): train/test split, default: 0.2/0.8      
        - seed (int): seed for the random number generator, default: 42
    Outputs:
        - train_df (pd.dataframe): training set
        - test_df (pd.dataframe): test set 
    """
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)     
    train_size = int(df.shape[0] * (1 - test_size))
    train_df = df.iloc[:train_size].reset_index(drop=True)
    test_df = df.iloc[train_size:].reset_index(drop=True)
    return train_df, test_df


def normalize(X, mean=None, std=None):
    if mean is None:
        mean = np.mean(X, axis=0)
    if std is None:
        std = np.std(X, axis=0)
    X_norm = (X - mean) / std
    return X_norm, mean, std


def one_hot_encoding(df, column):
    df = pd.concat([df, pd.get_dummies(df[column], prefix=column)], axis=1)
    df = df.drop(column, axis=1)
    return df


def mse(y1, y2):
    return np.mean((y1 - y2)**2)


def cross_validation_ridge(X, y, lmbdas, k=10, seed=42):
    """
    Perform k-fold cross validation for ridge regression
    Inputs:
        - X (np.array): input data
        - y (np.array): target data
        - lmbdas (list): list of regularization parameters
        - k (int): number of folds
        - seed (int): seed for the random number generator
    Outputs:
        - mse_train (np.array): training error for each fold and lambda
        - mse_test (np.array): test error for each fold and lambda
    """
    np.random.seed(seed)
    n = X.shape[0]
    mse_train = np.zeros((len(lmbdas), k))
    mse_test = np.zeros((len(lmbdas), k))

    for i, lmbda in enumerate(lmbdas):
        # idx = np.random.permutation(n)
        idx = np.arange(n)                                  # without shuffling
        X_shuffled = X.iloc[idx]  
        y_shuffled = y[idx]

        # print(f'x_shuffled[{i}.shape] = {X_shuffled.shape}')
        # print(f'y_shuffled[{i}.shape] = {y_shuffled.shape}')
        
        for j in range(k):
            X_train = np.concatenate([X_shuffled[:j*(n//k)], X_shuffled[(j+1)*(n//k):]])
            y_train = np.concatenate([y_shuffled[:j*(n//k)], y_shuffled[(j+1)*(n//k):]])
            X_test = X_shuffled[j*(n//k):(j+1)*(n//k)]
            y_test = y_shuffled[j*(n//k):(j+1)*(n//k)]
            
            lr_model = LinearRegression(lmbda)
            W = lr_model.fit(X_train, y_train)
            y_train_pred = lr_model.predict(X_train)
            y_test_pred = lr_model.predict(X_test)
            
            mse_train[i, j] = mse(y_train, y_train_pred)
            mse_test[i, j] = mse(y_test, y_test_pred)

    return mse_train, mse_test


class LinearRegression:

    def __init__(self, lmbda=0, verbose=False):
        self.lmbda = lmbda
        self.verbose = verbose
        self.W = None


    def fit(self, X, y):
        """
        Fits a linear regression model to the data
        Inputs:
            - X (np.array): input data
            - y (np.array): target data
            Outputs:
            - W (np.array): weights of the linear regression model
        """
        X = np.hstack((np.ones((X.shape[0], 1)), X))
        I = np.eye(X.shape[1])  # Identity matrix
        W = np.linalg.inv(X.T @ X + self.lmbda * I) @ X.T @ y
        self.W = W
        return W


    def predict(self, X):
        """
        Predicts the output of a linear regression model
        Inputs:
            - X (np.array): input data
            Outputs:
            - y_pred (np.array): predicted output
        """
        X = np.hstack((np.ones((X.shape[0], 1)), X))
        return X @ self.W
    

def k_folds_plot(lmbda, X, y, k=10):
    n = X.shape[0]

    for j in range(k):
        X_train = np.concatenate([X[:j*(n//k)], X[(j+1)*(n//k):]])
        y_train = np.concatenate([y[:j*(n//k)], y[(j+1)*(n//k):]])
        X_test = X[j*(n//k):(j+1)*(n//k)]
        y_test = y[j*(n//k):(j+1)*(n//k)]
        
        lr_model = LinearRegression(lmbda)
        W = lr_model.fit(X_train, y_train)
        y_test_pred = lr_model.predict(X_test)
        mse_test = mse(y_test, y_test_pred)

        # plot y_test vs y_test_pred
        plt.figure()
        plt.scatter(y_test, y_test_pred)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--')
        plt.title(f'Fold {j+1} Test Predictions')
        plt.xlabel('y_test')
        plt.ylabel('y_test_pred')
        plt.show()

        print(f'MSE for fold {j+1}: {mse_test}')