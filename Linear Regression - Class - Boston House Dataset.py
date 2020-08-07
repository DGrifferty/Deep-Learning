import numpy as np
from numpy import ndarray
from typing import Callable, Dict, Tuple, List
from sklearn.datasets import load_boston
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt
import time
import math


class LinearRegression:

    def __init__(self, X: ndarray, y: ndarray, split: float = 0.25):

        self.y = y
        self.X = X
        self.split = split

        r = np.c_[self.X.reshape(len(self.X), -1), self.y.reshape(len(self.y), -1)]
        self.X = r[:, :self.X.size // len(self.X)].reshape(X.shape)
        self.y = r[:, self.X.size // len(self.X):].reshape(-1, 1)

        np.random.shuffle(r)

        self.X_train = self.X[0: round(len(self.X) * self.split), :]
        self.y_train = self.y[0: round(len(self.X) * self.split), :]

        self.X_test = self.X[0: round(len(self.X) * self.split), :]
        self.y_test = self.y[0: round(len(self.X) * self.split), :]

        self.weights = dict()
        self.weights['W'] = np.random.rand(self.X.shape[1], 1)
        self.weights['B'] = np.random.randn(1, 1)

        self.learning_rate = 0.001
        self.epochs = 1000
        self.batch_size = 40  # Make variable for different data sizes

        self.__repr__()

    def __repr__(self):
        pass

    def _move_forward(self, **kwargs):
        X_batch = kwargs.get('X_batch', self.X_train)
        y_batch = kwargs.get('y_batch', self.y_train)
        weights = kwargs.get('weights', self.weights)

        # X_batch: ndarray = self.X_train, y_batch: ndarray = self.y_train,
        # weights: Dict[str, ndarray] = self.weights) -> Tuple[float, Dict[str, ndarray]]:

        assert X_batch.shape[0] == y_batch.shape[0]
        assert X_batch.shape[1] == weights['W'].shape[0]
        assert weights['B'].shape[0] == weights['B'].shape[1] == 1

        N = np.dot(X_batch, weights['W'])
        Pred = N + weights['B']

        # loss = np.mean(np.power(Pred - y_batch, 2))

        forward: Dict[str, ndarray] = dict()
        forward['X'] = X_batch
        forward['Y'] = y_batch
        forward['N'] = N
        forward['P'] = Pred

        self.forward = forward

        print('completed')

        return forward

    def mae(y_pred: ndarray, y: ndarray) -> float:
        return np.mean(np.abs(y - y_pred))

    def rmse(y_pred: ndarray, y: ndarray) -> float:
        return np.sqrt(np.mean(np.power(y - y_pred, 2)))

    def _loss_grads(self, forward: Dict[str, ndarray], weights: Dict[str, ndarray]) -> Dict[str, ndarray]:
        lossgrad = dict()

        dLdP = -2 * (forward['Y'] - forward['P'])
        dPdN = np.ones_like(forward['N'])
        dPdB = np.ones_like(weights['B'])
        dLdN = dLdP * dPdN
        dNdW = np.transpose(forward['X'], (1, 0))
        dLdW = np.dot(dNdW, dLdN)
        dLdB = (dLdP * dPdB).sum(axis=0)

        lossgrad['W'] = dLdW
        lossgrad['B'] = dLdB

        self.lossgrad = lossgrad

        return lossgrad

    def random_batch(X: ndarray, y: ndarray, batch_size: int) -> Tuple[ndarray, ndarray]:
        assert X.ndim == y.ndim == 2

        r = np.c_[X.reshape(len(X), -1), y.reshape(len(y), -1)]
        X = r[:, :X.size // len(X)].reshape(X.shape)
        y = r[:, X.size // len(X):].reshape(-1, 1)

        np.random.shuffle(r)

        X_batch = X[0: batch_size, :]
        y_batch = y[0: batch_size, :]

        return X_batch, y_batch

    def fit(self, **kwargs):
        X = kwargs.get('X', self.X_train)
        y = kwargs.get('y', self.y_train)
        weights = kwargs.get('weights', self.weights)
        learning_rate = kwargs.get('learning_rate', self.learning_rate)
        epochs = kwargs.get('epochs', self.epochs)

        for i in range(epochs):
            X_batch, y_batch = random_batch(X, y, round(0.1 * len(X)))
            move_forward(X_batch, y_batch, weights)
            loss_grads(self.forward, weights)

            for key in weights.keys():
                weights[key] -= learning_rate * self.lossgrads[key]

        self.weights = weights

        return weights




boston = load_boston()
data = boston.data
target = boston.target
features = boston.feature_names

s = StandardScaler()
data = s.fit_transform(data)

test = LinearRegression(data, target)
test._move_forward()
test.fit()
