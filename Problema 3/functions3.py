import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from tqdm import tqdm


def train_test_split(df, test_size=0.2, seed=42):
    # shuffle dataset con una semilla fija
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)     
    train_size = int(df.shape[0] * (1 - test_size))
    train_df = df.iloc[:train_size].reset_index(drop=True)
    test_df = df.iloc[train_size:].reset_index(drop=True)
    return train_df, test_df


def one_hot_encoding(df, column):
    df = pd.concat([df, pd.get_dummies(df[column], prefix=column)], axis=1)
    df = df.drop(column, axis=1)
    return df


def set_weights_and_biases(l=[1, 3, 1], activations=['relu', 'linear'], seed=42 ):
    """
    input:
        L: profundidad de la red (cantidad de capas ocultas)
        M^l: cantidad de unidades ocultas en la capa l 
    """
    # check consistency of the input
    L = len(l)
    assert len(activations) + 1 == L, 'Debe haber una función de activación por capa'

    np.random.seed(seed)
    # initialize weights and biases
    W, b = {}, {}
    for i in range(1, L):  
        W[i] = np.random.randn(l[i-1], l[i]) 
        b[i] = np.random.randn(l[i])
    return W, b


def forward_pass(X, W, b, L, activations=['relu', 'linear']):
    """
    input:
        X: matriz de diseño
        W: pesos de la red
        b: sesgos de la red
        L: profundidad de la red (cantidad de capas ocultas)
        activations: lista con las funciones de activación de cada capa
    """
    # L = len(W) + 1                                                      # CHEQUEAR

    # initialize the dictionary to store the activations
    A = {0: X}
    Z = {}
    for i in range(1, L):
        Z[i] = A[i-1] @ W[i] + b[i]
    
        if activations[i-1] == 'relu':
            A[i] = np.maximum(Z[i], 0)
        elif activations[i-1] == 'linear':
            A[i] = Z[i]
        elif activations[i-1] == 'sigmoid':
            A[i] = 1 / (1 + np.exp(-Z[i]))

    return A, Z


def backward_pass(X, y, W, b, A, Z, L, activations=['relu', 'linear']):
    """
    input:
        X: matriz de diseño
        y: vector de etiquetas
        W: pesos de la red
        b: sesgos de la red
        A: diccionario con las activaciones
        Z: diccionario con las salidas de las capas ocultas
        L: profundidad de la red (cantidad de capas ocultas)
        activations: lista con las funciones de activación de cada capa
    """
    # L = len(W) + 1                                                      # CHEQUEAR

    # initialize the dictionary to store the gradients
    dW, db = {}, {}
    dZ = {}

    dA = {L-1: -2 * (y - A[L-1])}
    for i in range(L-1, 0, -1):
        if activations[i-1] == 'relu':
            dZ[i] = dA[i] * (Z[i] > 0)
        elif activations[i-1] == 'linear':
            dZ[i] = dA[i]
        elif activations[i-1] == 'sigmoid':
            dZ[i] = dA[i] * A[i] * (1 - A[i])

        dW[i] = A[i-1].T @ dZ[i]
        db[i] = np.sum(dZ[i], axis=0)
        dA[i-1] = dZ[i] @ W[i].T

    return dW, db


def update_weights_and_biases(W, b, dW, db, alpha=0.01):
    """
    input:
        W: pesos de la red
        b: sesgos de la red
        dW: gradientes de los pesos
        db: gradientes de los sesgos
        alpha: learning rate
    """
    for i in W.keys():
        W[i] -= alpha * dW[i]
        b[i] -= alpha * db[i]
    return W, b


def fit_nn(X, y, l=[1, 3, 1], activations=['relu', 'linear'], alpha=0.01, max_epoch=1000, seed=42):
    """
    input:
        X: matriz de diseño
        y: vector de etiquetas
        L: profundidad de la red (cantidad de capas ocultas)
        M^l: cantidad de unidades ocultas en la capa l 
        activations: lista con las funciones de activación de cada capa
        alpha: learning rate
        max_epoch: cantidad máxima de iteraciones
    """
    L = len(l)
    # initialize weights and biases
    W, b = set_weights_and_biases(l, activations, seed)
    for epoch in tqdm(range(max_epoch)):
        # forward pass
        A, Z = forward_pass(X, W, b, L, activations)
        # backward pass
        dW, db = backward_pass(X, y, W, b, A, Z, L, activations)
        # update weights and biases
        W, b = update_weights_and_biases(W, b, dW, db, alpha)
    return W, b

def predict_nn(X, W, b, activations=['relu', 'linear']):
    """
    input:
        X: matriz de diseño
        W: pesos de la red
        b: sesgos de la red
        activations: lista con las funciones de activación de cada capa
    """
    L = len(W) + 1
    A, Z = forward_pass(X, W, b, L, activations)
    return A[L-1]