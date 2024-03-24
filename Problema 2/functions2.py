import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def train_test_split(df, test_size=0.2, seed=42):
    # shuffle dataset con una semilla fija
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)     
    train_size = int(df.shape[0] * (1 - test_size))
    train_df = df.iloc[:train_size].reset_index(drop=True)
    test_df = df.iloc[train_size:].reset_index(drop=True)
    return train_df, test_df


# def fit_linear_regression(X, y, M):
#     # Generate polynomial features
#     Phi = np.ones((X.shape[0], 1)) 
#     for i in range(1, M+1):
#         Phi = np.c_[Phi, X**i]
    
#     # Compute weights using the normal equation
#     W = np.linalg.inv(Phi.T @ Phi) @ Phi.T @ y
    
#     return W


def predict_linear_regression(X, w):
    X = np.hstack((np.ones((X.shape[0], 1)), X)) 
    return X @ w


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


def fit_ridge_regression(X, y, lmbda):
    X = np.hstack((np.ones((X.shape[0], 1)), X))
    # Generate polynomial features
    I = np.eye(X.shape[1])  # Identity matrix
    W = np.linalg.inv(X.T @ X + lmbda * I) @ X.T @ y
    return W


def mse(y1, y2):
    return np.mean((y1 - y2)**2)


def cross_validation_ridge(X, y, lmbdas, k=10, seed=42):
    np.random.seed(seed)
    n = X.shape[0]
    mse_train = np.zeros((len(lmbdas), k))
    mse_test = np.zeros((len(lmbdas), k))
    
    for i, lmbda in enumerate(lmbdas):
        idx = np.random.permutation(n)
        X_shuffled = X.iloc[idx]  
        y_shuffled = y[idx]
        for j in range(k):
            X_train = np.concatenate([X_shuffled[:j*(n//k)], X_shuffled[(j+1)*(n//k):]])
            y_train = np.concatenate([y_shuffled[:j*(n//k)], y_shuffled[(j+1)*(n//k):]])
            X_test = X_shuffled[j*(n//k):(j+1)*(n//k)]
            y_test = y_shuffled[j*(n//k):(j+1)*(n//k)]
            
            W = fit_ridge_regression(X_train, y_train, lmbda)
            y_train_pred = predict_linear_regression(X_train, W)
            y_test_pred = predict_linear_regression(X_test, W)
            
            mse_train[i, j] = mse(y_train, y_train_pred)
            mse_test[i, j] = mse(y_test, y_test_pred)

        # # grafico mse_train y mse_test en mismo grafico
        # plt.figure()
        # plt.plot(mse_train[i], label='train')
        # plt.plot(mse_test[i], label='test')
        # plt.title(f'MSE for lambda={lmbda}')
        # plt.xlabel('Fold')
        # plt.ylabel('MSE')
        # plt.legend()
        # plt.show()

    return mse_train, mse_test