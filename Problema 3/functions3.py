import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from tqdm import tqdm


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


def one_hot_encoding(df, column):
    df = pd.concat([df, pd.get_dummies(df[column], prefix=column)], axis=1)
    df = df.drop(column, axis=1)
    return df


class MLP(object):

    def __init__(self, input_size, layers=[6, 30, 1], activations='default', seed=42, verbose=False):
        self.verbose = verbose
        self.seed = seed
        self.input_size = input_size
        self.layers = layers  
        self.num_layers = len(self.layers)
        if activations == 'default':
            self.activations = ['relu'] * (self.num_layers -1) + ['linear']
        else: self.activations = activations
        self.check_compatability()
        self.set_weights_and_biases()

    def check_compatability(self):
        assert len(self.activations) == self.num_layers, 'Debe haber una función de activación por capa'


    def set_weights_and_biases(self):
        """
        Initialize weights and biases randomly
        """
        np.random.seed(self.seed)
        self.biases = [np.random.randn(y, 1) for y in self.layers[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(self.layers[:-1], self.layers[1:])]
        if self.verbose:
            print(f"b.shape: {self.biases[0].shape}")
            print(f"W.shape: {self.weights[0].shape}")


    def activation_function(self, activation_str):
        """
        Returns the activation function given its name
        
        Inputs:
            - activation_str (str): name of the activation function
        Outputs:
            - lambda function: activation function
        """
        if activation_str == 'relu':
            return lambda z : np.maximum(z, 0)
        elif activation_str == 'linear':
            return lambda z : z
        elif activation_str == 'sigmoid':
            return lambda z : 1 / (1 + np.exp(-z))
        else:
            print("Invalid activation function")
        

    def deriv_activation_function(self, activation_str):
        """
        Returns the derivative of the activation function given its name

        Inputs:
            - activation_str (str): name of the activation function
        Outputs:
            - lambda function: derivative of the activation function
        """
        if activation_str == 'relu':
            return lambda z : (z > 0).astype(int)
        elif activation_str == 'linear':
            return lambda z : np.ones(z.shape)
        elif activation_str == 'sigmoid':
            return lambda z : z * (1 - z)
        else:
            print("Invalid activation function")


    def compute_loss(self, a_out, y):
        return np.mean((a_out - y) ** 2)


    def forward_pass(self, X):
        """
        Forward pass through the network
        
        Inputs:
            - X (np.array): input data
        Outputs:
            - a (list): list of activations
            - z (list): list of weighted inputs
        """
        z = [np.array(X).reshape(-1, 1)]
        a = []
        for l in range(1, self.num_layers):
            a_l = np.dot(self.weights[l-1], z[l-1]) + self.biases[l-1]
            a.append(np.copy(a_l))

            # Check if the current layer has an associated activation function
            if l < len(self.activations):
                h = self.activation_function(self.activations[l-1])
                z_l = h(a_l)
            else:
                # If no activation function is specified, use linear activation
                z_l = a_l
                # # If no activation function is specified, use relu activation
                # z_l = np.maximum(a_l, 0)

            z.append(np.copy(z_l))

        if self.verbose:
            print(f"z.shape: {z[0].shape}", end=" ")
            print(f"a.shape: {a[0].shape}", end=" ")

        return a, z


    def backward_pass(self, a, z, y):
        """
        Backward pass through the network

        Inputs:
            - a (list): list of activations
            - z (list): list of weighted inputs
            - y (np.array): target data
        Outputs:
            - loss (float): loss of the network
            - nabla_w (list): list of gradients of the weights
            - nabla_b (list): list of gradients of the biases
        """ 
        d = [np.zeros(w.shape) for w in self.weights]
        h_deriv = self.deriv_activation_function(self.activations[-1])
        d[-1] = (a[-1] - y) * h_deriv(a[-1])

        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_b[-1] = d[-1]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        nabla_w[-1] = np.dot(d[-1], z[-2].T)

        for l in reversed(range(1, len(d))):
            h_deriv = self.deriv_activation_function(self.activations[l-1])
            d[l-1] = np.dot(self.weights[l].T, d[l]) * h_deriv(a[l-1])
            nabla_b[l-1] = d[l-1]
            nabla_w[l-1] = np.dot(d[l-1], z[l-1].T)
        
        loss = self.compute_loss(a[-1], y)
        return loss, nabla_w, nabla_b


    def update_mini_batch(self, mini_batch, alpha):
        """
        Update weights and biases using mini-batch gradient descent
        
        Inputs:
            - mini_batch (list): list of mini-batches
            - alpha (float): learning rate
            Outputs:
            - total_loss (float): total loss of the mini-batch
        """
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        total_loss = 0

        for x, y in mini_batch:
            a, z = self.forward_pass(x)
            loss, d_nabla_w, d_nabla_b = self.backward_pass(a, z, y)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, d_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, d_nabla_w)]
            total_loss += loss

        self.weights = [w - (alpha / len(mini_batch)) * nw for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (alpha / len(mini_batch)) * nb for b, nb in zip(self.biases, nabla_b)]
        return total_loss


    def update_single_example(self, x, y, alpha):
        """
        Update weights and biases using single example gradient descent
        
        Inputs:
            - x (np.array): input data
            - y (np.array): target data
            - alpha (float): learning rate
            Outputs:
            - loss (float): loss of the single example
        """
        a, z = self.forward_pass(x)
        loss, nabla_w, nabla_b = self.backward_pass(a, z, y)
        self.weights = [w - alpha * nw for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - alpha * nb for b, nb in zip(self.biases, nabla_b)]
        return loss


    def evaluate(self, test_data):
        """
        Evaluate the network on test data

        Inputs:
            - test_data (list): list of test data
        Outputs:
            - sum_sq_error (float): mean squared error of the test data
        """
        sum_sq_error = 0
        for x, y in test_data:
            pred = self.forward_pass(x)[-1][-1].flatten()
            sum_sq_error += self.compute_loss(pred, y)
        return sum_sq_error / len(test_data)


    def fit(self, training_data, test_data, mini_batch_size, alpha=0.01, max_epochs=100, update_rule='mini_batch'):
        """
        Fit the model to the training data

        Inputs:
            - training_data (list): list of training data
            - test_data (list): list of test data
            - mini_batch_size (int): size of the mini-batches
            - alpha (float): learning rate
            - max_epochs (int): maximum number of epochs
            - update_rule (str): update rule for the network
        Outputs:
            - train_losses (list): list of training losses
            - test_losses (list): list of test losses
        """
        if update_rule == 'mini_batch':
            train_losses, test_losses = [], []
            n_train = len(training_data)

            for epoch in tqdm(range(max_epochs)):
                # random.shuffle(training_data)
                mini_batches = [training_data[k:k+mini_batch_size] for k in range(0, n_train, mini_batch_size)]

                for mini_batch in mini_batches:
                    train_loss = self.update_mini_batch(mini_batch, alpha)
                
                train_losses.append(train_loss)
    
                test_loss = self.evaluate(test_data)
                test_losses.append(test_loss)

                if self.verbose:
                    print(f"Epoch {epoch}: Train loss: {train_loss}, Test loss: {test_loss}")

            return train_losses, test_losses
        
        elif update_rule == 'single_example':
            train_losses, test_losses = [], []

            for epoch in tqdm(range(max_epochs)):
                # random.shuffle(training_data)

                for x, y in training_data:
                    train_loss = self.update_single_example(x, y, alpha)
                
                train_losses.append(train_loss)
                
                test_loss = self.evaluate(test_data)
                test_losses.append(test_loss)

                if self.verbose:
                    print(f"Epoch {epoch}: Train loss: {train_loss}, Test loss: {test_loss}")
                
            return train_losses, test_losses
    

    def predict(self, X):
        X = X.values
        predictions = []
        for x in X:
            a, z = self.forward_pass(x.reshape(-1, 1))
            pred = z[-1][-1].flatten()
            predictions.append(pred)

        return np.array(predictions)
    

def k_folds_plot(X, y, k=10, layers=[6, 30, 1]):
    n = X.shape[0]

    for j in range(k):
        X_train = np.concatenate([X[:j*(n//k)], X[(j+1)*(n//k):]])
        y_train = np.concatenate([y[:j*(n//k)], y[(j+1)*(n//k):]])
        X_test = X[j*(n//k):(j+1)*(n//k)]
        y_test = y[j*(n//k):(j+1)*(n//k)]
        
        lr_model = MLP(input_size=X_train.shape[1], layers=layers)
        W = lr_model.fit(X_train, y_train)
        y_test_pred = lr_model.predict(X_test)
        mse_test = np.mean((y_test - y_test_pred)**2)

        # plot y_test vs y_test_pred
        plt.figure()
        plt.scatter(y_test, y_test_pred)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--')
        plt.title(f'Fold {j+1} Test Predictions')
        plt.xlabel('y_test')
        plt.ylabel('y_test_pred')
        plt.show()

        print(f'MSE for fold {j+1}: {mse_test}')