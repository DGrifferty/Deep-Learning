import numpy as np
from numpy import ndarray
from typing import Callable, Dict, Tuple, List
from sklearn.datasets import load_boston
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt
import time
import math


class LinearRegression(X, y, split):

    def __init__(self):

        self.y == y
        self.X == X
        self.split = split

        r = np.c_[self.X.reshape(len(self.X), -1), self.y.reshape(len(self.y), -1)]
        self.X = r[:, :self.X.size // len(self.X)].reshape(X.shape)
        self.y = r[:, self.X.size // len(self.X):].reshape(-1, 1)

        np.random.shuffle(r)

        self.X_train = X[0: round(len(self.X) * self.split), :]
        self.y_train = y[0: round(len(self.X) * self.split), :]

        self.X_test = X[0: round(len(self.X) * self.split), :]
        self.y_test = y[0: round(len(self.X) * self.split), :]

        self.weights = dict()
        self.weights['W'] = np.random.rand(self.X.shape[1], 1)
        self.weights['B'] = np.random.randn(1, 1)

    def move_forward(self, X_batch: ndarray = self.X_train, y_batch: ndarray = self.y_train,
                     weights: Dict[str, ndarray] = self.weights) -> Tuple[float, Dict[str, ndarray]]:

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

        return forward

    def mae(y_pred: ndarray, y: ndarray) -> float:
        return np.mean(np.abs(y - y_pred))

    def rmse(y_pred: ndarray, y: ndarray) -> float:
        return np.sqrt(np.mean(np.power(y - y_pred, 2)))

    def loss_grads(forward: Dict[str, ndarray], weights: Dict[str, ndarray]) -> Dict[str, ndarray]:
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

        return lossgrad

    def fit(self, learning_rate: float = learning_rate_default, epochs: int = epochs_default):
        pass



