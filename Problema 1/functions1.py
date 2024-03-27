import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tqdm


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


def fit_linear_regression(X, y, M, verbose=False):
    """
    Fits a polynomial regression model of degree M to the data
        Phi: matrix initialized with ones in the first column and polynomial features in the following columns
    Inputs:
        - X (np.array): input data                                    CHEQUEAR
        - y (np.array): target data                                   CHEQUEAR
        - M: degree of the polynomial
    Outputs:
        - W (np.array): weights of the polynomial regression model
    """
    # Generate polynomial features
    Phi = np.ones((X.shape[0], 1)) 
    for i in range(1, M+1):
        Phi = np.c_[Phi, X**i]

    # Compute weights using the normal equation
    W = np.linalg.inv(Phi.T @ Phi) @ Phi.T @ y
    
    if verbose:
        print(f'Phi shape: {Phi.shape}')
        print(f'W shape: {W.shape}')
        print(f'Phi: {Phi}')
        print(f'W: {W}')
    
    return W


def predict_linear_regression(X, w):
    """
    Predicts the output of a linear regression model
    Inputs:
        - X (np.array): input data
        - w (np.array): weights of the model
    Outputs:
        - y_pred (np.array): predicted output
    """
    poly = np.polynomial.polynomial.Polynomial(w)   
    return poly(X)


def fit_ridge_regression(X, y, M, lmbda, verbose=False):
    """
    Fits a polynomial ridge regression model of degree M to the data
    Inputs:
        - X (np.array): input data
        - y (np.array): target data
        - M: degree of the polynomial
        - lmbda: regularization parameter
    Outputs:
        - W (np.array): weights of the polynomial ridge regression model
    """
    # Generate polynomial features
    Phi = np.ones((X.shape[0], 1)) 
    for i in range(1, M+1):
        Phi = np.c_[Phi, X**i]
    
    # Compute weights using the normal equation
    W = np.linalg.inv(Phi.T @ Phi + lmbda * np.eye(M+1)) @ Phi.T @ y

    if verbose:
        print(f'Phi shape: {Phi.shape}')
        print(f'W shape: {W.shape}')
        print(f'Phi: {Phi}')
        print(f'W: {W}')
    
    return W
