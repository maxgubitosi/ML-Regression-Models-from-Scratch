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


# def set_weights_and_biases(l=[1, 6, 1], activations=['relu', 'linear'], seed=42 ):
#     """
#     input:
#         L: profundidad de la red (cantidad de capas ocultas)
#         M^l: cantidad de unidades ocultas en la capa l 
#     """
#     # check consistency of the input
#     L = len(l)
#     assert len(activations) + 1 == L, 'Debe haber una función de activación por capa'

#     np.random.seed(seed)
#     # initialize weights and biases
#     W, b = {}, {}
#     b = {i: np.random.randn(j, 1) for i, j in enumerate(l[1:], start=1)}
#     W = {i: np.random.randn(y, x) for i, (x, y) in enumerate(zip(l[:-1], l[1:]), start=1)}

#     # prints para debug
#     print(f"b.shape: {b[1].shape}")
#     print(f"W.shape: {W[1].shape}")

#     print(f"b: {b}")
#     print(f"W: {W}")
#     return W, b


# def activation_function(z, activation):
#     if activation == 'relu':
#         return np.maximum(z, 0)
#     elif activation == 'linear':
#         return z
#     elif activation == 'sigmoid':
#         return 1 / (1 + np.exp(-z))

# def compute_loss(A_out, y):
#     return np.mean((A_out - y) ** 2)


# def forward_pass(X, W, b, L, activations=['relu', 'linear']):
#     """
#     input:
#         X: matriz de diseño
#         W: pesos de la red
#         b: sesgos de la red
#         L: profundidad de la red (cantidad de capas ocultas)
#         activations: lista con las funciones de activación de cada capa
#     """
#     # # L = len(W) + 1                                                      # CHEQUEAR

#     # # initialize the list to store the activations
#     # A = [X]
#     # Z = []
#     # for i in range(1, L):  # Corrección en el rango de la iteración
#     #     Z.append(A[i-1] @ W[i] + b[i])  # Añadir las salidas de la capa oculta
#     #     A.append(activation_function(Z[i-1], activations[i-1]))  # Añadir las activaciones
#     # return A, Z

#     z = [np.array(X).reshape(-1, 1)]
#     a = [z[0] for _ in range(L)]  # Inicializa 'a' con la misma forma que 'z'
#     for i in range(1, L):
#         z.append(np.dot(W[i], a[i-1]) + b[i])
#         a[i] = activation_function(z[i], activations[i-1])
#     return a, z

# def backward_pass(X, y, W, b, A, Z, L, activations=['relu', 'linear']):
#     """
#     input:
#         X: matriz de diseño
#         y: vector de etiquetas
#         W: pesos de la red
#         b: sesgos de la red
#         A: diccionario con las activaciones
#         Z: diccionario con las salidas de las capas ocultas
#         L: profundidad de la red (cantidad de capas ocultas)
#         activations: lista con las funciones de activación de cada capa
#     """
#     # L = len(W) + 1                                                      # CHEQUEAR

#     # initialize the dictionary to store the gradients
#     dW, db = {}, {}
#     dZ = {}

#     dA = {L-1: -2 * (y - A[L-1])}
#     for i in range(L-1, 0, -1):
#         if activations[i-1] == 'relu':
#             dZ[i] = dA[i] * (Z[i] > 0)
#         elif activations[i-1] == 'linear':
#             dZ[i] = dA[i]
#         elif activations[i-1] == 'sigmoid':
#             dZ[i] = dA[i] * A[i] * (1 - A[i])

#         dW[i] = A[i-1].T @ dZ[i]
#         db[i] = np.sum(dZ[i], axis=0)
#         dA[i-1] = dZ[i] @ W[i].T

#     return dW, db


# def update_weights_and_biases(W, b, dW, db, alpha=0.01):
#     """
#     input:
#         W: pesos de la red
#         b: sesgos de la red
#         dW: gradientes de los pesos
#         db: gradientes de los sesgos
#         alpha: learning rate
#     """
#     for i in W.keys():
#         W[i] -= alpha * dW[i]
#         b[i] -= alpha * db[i]
#     return W, b


# def fit_nn(X, y, l=[1, 3, 1], activations=['relu', 'linear'], alpha=0.01, max_epoch=1000, seed=42):
#     """
#     input:
#         X: matriz de diseño
#         y: vector de etiquetas
#         L: profundidad de la red (cantidad de capas ocultas)
#         M^l: cantidad de unidades ocultas en la capa l 
#         activations: lista con las funciones de activación de cada capa
#         alpha: learning rate
#         max_epoch: cantidad máxima de iteraciones
#     """
#     L = len(l)
#     # initialize weights and biases
#     W, b = set_weights_and_biases(l, activations, seed)
#     for epoch in tqdm(range(max_epoch)):
#         # forward pass
#         A, Z = forward_pass(X, W, b, L, activations)
#         # backward pass
#         dW, db = backward_pass(X, y, W, b, A, Z, L, activations)
#         # update weights and biases
#         W, b = update_weights_and_biases(W, b, dW, db, alpha)
#     return W, b

# def predict_nn(X, W, b, activations=['relu', 'linear']):
#     """
#     input:
#         X: matriz de diseño
#         W: pesos de la red
#         b: sesgos de la red
#         activations: lista con las funciones de activación de cada capa
#     """
#     L = len(W) + 1
#     A, Z = forward_pass(X, W, b, L, activations)
#     return A[L-1]


# class MLP(object):
#     def __init__(self, layers=[1, 3, 1], activations=['relu', 'linear']):
#         self.layers = layers
#         self.activations = activations
#         self.num_layers = len(layers)

#         self.biases = [np.random.randn(y,1) for y in layers[1:]]
#         self.weights = [np.random.randn(y,x) for x,y in zip(layers[:-1], layers[1:])]


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

class MLP(object):

    def __init__(self, layers=[1, 30, 1], activations=['relu', 'linear'], seed=42, debug=False):
        self.debug = debug
        self.seed = seed
        self.layers = layers
        self.activations = activations
        self.num_layers = len(layers)
        self.set_weights_and_biases()


    def set_weights_and_biases(self):
        np.random.seed(self.seed)
        self.biases = [np.random.randn(y, 1) for y in self.layers[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(self.layers[:-1], self.layers[1:])]
        if self.debug:
            print(f"b.shape: {self.biases[0].shape}")
            print(f"W.shape: {self.weights[0].shape}")


    def activation_function(activation_str):
        if activation_str == 'relu':
            return lambda z : np.maximum(z, 0)
        elif activation_str == 'linear':
            return lambda z : z
        elif activation_str == 'sigmoid':
            return lambda z : 1 / (1 + np.exp(-z))
        

    def deriv_activation_function(activation_str):
        if activation_str == 'relu':
            return lambda z : (z > 0).astype(int)
        elif activation_str == 'linear':
            return lambda z : np.ones(z.shape)
        elif activation_str == 'sigmoid':
            return lambda z : z * (1 - z)


    def compute_loss(a_out, y):
        return np.mean((a_out - y) ** 2)


    def forward_pass(self, X):
        z = [np.array(X).reshape(-1, 1)]
        a = []
        for l in range(1, self.num_layers):
            a_l = np.dot(self.weights[l-1], z[l-1]) + self.biases[l-1]
            a.append(np.copy(a_l))

            h = self.activation_function(a_l, self.activations[l-1])
            z_l = h(a_l)
            z.append(np.copy(z_l))

        if self.debug:
            print(f"z.shape: {z.shape}")
            print(f"a.shape: {a.shape}")

        return a, z


    def backward_pass(self, a, z, y):
        d = [np.zeros(w.shape) for w in self.weights]
        h_deriv = self.deriv_activation_function(self.activations[-1])
        d[-1] = (a[-1] - y) * h_deriv(a[-1])

        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_b[-1] = d[-1]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        nabla_w[-1] = np.dot(d[-1], z[-2].T)

        for l in range(1, self.num_layers):
            h_deriv = self.deriv_activation_function(self.activations[l-1])
            d[l-1] = np.dot(self.weights[l].T, d[l]) * h_deriv(a[l-1])
            nabla_b[l-1] = d[l-1]
            nabla_w[l-1] = np.dot(d[l-1], z[l-1].T)
        
        loss = self.compute_loss(a[-1], y)
        return loss, nabla_w, nabla_b


    def update_mini_batch(self, mini_batch, alpha):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        total_loss = 0

        for x, y in mini_batch:
            a, z = self.forward_pass(x)
            loss, d_nabla_w, d_nabla_b = self.backward_pass(a, z, y)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, d_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, d_nabla_w)]
            total_loss += loss

        self.weights = [w - (alpha / len(mini_batch)) * nw for w, nw in zip(self.weights, nabla_w)]  # CHEQUEAR (ultimas 3 lineas) !!!!!!!!!!!!!!!!!!!! 
        self.biases = [b - (alpha / len(mini_batch)) * nb for b, nb in zip(self.biases, nabla_b)]
        return total_loss


    def evaluate(self, X_test, y_test):
        sum_sq_error = 0
        for x, y in zip(X_test, y_test):    # CHEQUEAR si no es zip(*X_test, y_test) o algo asi
            pred = self.forward_pass(x)[-1][-1]
            sum_sq_error += self.compute_loss(pred, y)
        return sum_sq_error / len(X_test)


    def fit(self, X_train, y_train, X_test, y_test, mini_batch_size, alpha=0.01, max_epochs=100):
        train_losses, test_losses = [], []
        n_train = len(X_train)

        for epoch in tqdm(range(max_epochs)):
            random.shuffle(X_train)
            mini_batches = [X_train[k:k+mini_batch_size] for k in range(0, n_train, mini_batch_size)]

            for mini_batch in mini_batches:
                train_loss = self.update_mini_batch(mini_batch, alpha)
            
            train_losses.append(train_loss)
            
            test_loss = self.evaluate(X_test, y_test)
            test_losses.append(test_loss)

            if self.debug:
                print(f"Epoch {epoch}: Train loss: {train_loss}, Test loss: {test_loss}")
            
        return train_losses, test_losses
            

    def predict(self, X):
        pass  # Implement the prediction method
