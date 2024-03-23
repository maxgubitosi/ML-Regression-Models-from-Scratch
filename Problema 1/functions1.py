import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tqdm

def train_test_split(df, test_size=0.2, seed=42):
    # shuffle dataset con una semilla fija
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)     
    train_size = int(df.shape[0] * (1 - test_size))
    train_df = df.iloc[:train_size].reset_index(drop=True)
    test_df = df.iloc[train_size:].reset_index(drop=True)
    return train_df, test_df

def fit_linear_regression(X, y, M):
    # Generate polynomial features
    Phi = np.ones((X.shape[0], 1)) 
    for i in range(1, M+1):
        Phi = np.c_[Phi, X**i]
    
    # Compute weights using the normal equation
    W = np.linalg.inv(Phi.T @ Phi) @ Phi.T @ y
    
    return W

def predict_linear_regression(X, w):
    poly = np.polynomial.polynomial.Polynomial(w)   
    return poly(X)

