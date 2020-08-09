import numpy as np
from numpy import ndarray
from typing import Callable, Dict, Tuple, List
from sklearn.datasets import load_boston
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import time
import math


def boston_data() -> Tuple[ndarray, ndarray]:
    boston = load_boston()
    data = boston.data
    target = boston.target
    s = StandardScaler()
    data = s.fit_transform(data)

    return data, target


class LinearRegression:

    def __init__(self, X: ndarray, y: ndarray, split: float = 0.25):

        self.y = y
        self.X = X
        self.split = split

        r = np.c_[self.X.reshape(len(self.X), -1),
                  self.y.reshape(len(self.y), -1)]
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

    def move_forward(self, **kwargs):
        X_batch = kwargs.get('X_batch', self.X_train)
        y_batch = kwargs.get('y_batch', self.y_train)
        weights = kwargs.get('weights', self.weights)

        assert X_batch.shape[0] == y_batch.shape[0]
        assert X_batch.shape[1] == weights['W'].shape[0]
        assert weights['B'].shape[0] == weights['B'].shape[1] == 1

        N = np.dot(X_batch, weights['W'])
        pred = N + weights['B']

        # loss = np.mean(np.power(Pred - y_batch, 2))

        forward: Dict[str, ndarray] = dict()
        forward['X'] = X_batch
        forward['Y'] = y_batch
        forward['N'] = N
        forward['P'] = pred

        self.forward = forward

        return forward

    @staticmethod
    def mae(y_pred: ndarray, y: ndarray) -> ndarray:
        return np.mean(np.abs(y - y_pred))

    @staticmethod
    def rmse(y_pred: ndarray, y: ndarray) -> float:
        return np.sqrt(np.mean(np.power(y - y_pred, 2)))

    def loss_grads(self, **kwargs):
        forward = kwargs.get('forward', self.forward)
        weights = kwargs.get('weights', self.weights)

        dLdP = -2 * (forward['Y'] - forward['P'])
        dPdN = np.ones_like(forward['N'])
        dPdB = np.ones_like(weights['B'])
        dLdN = dLdP * dPdN
        dNdW = np.transpose(forward['X'], (1, 0))
        dLdW = np.dot(dNdW, dLdN)
        dLdB = (dLdP * dPdB).sum(axis=0)

        lossgrad = dict()
        lossgrad['W'] = dLdW
        lossgrad['B'] = dLdB

        self.lossgrad = lossgrad

        return lossgrad

    def random_batch(self, **kwargs):

        X = kwargs.get('X', self.X_train)
        y = kwargs.get('y', self.y_train)
        batch_size = kwargs.get('batch_size', self.batch_size)

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
            X_batch, y_batch = self.random_batch(
                X=X,
                y=y,
                batch_size=round(0.1 * len(X))
            )

            self.move_forward(X=X_batch, y=y_batch, weights=weights)
            self.loss_grads(forward=self.forward, weights=weights)

            for key in weights.keys():
                weights[key] -= learning_rate * self.lossgrad[key]

        self.weights = weights

        return weights

    def predict_compare(self, **kwargs):
        X = kwargs.get('X', self.X_test)
        y = kwargs.get('y', self.y_test)
        weights = kwargs.get('weights', self.weights)

        pred = np.dot(X, weights['W']) + weights['B']

        print(f'Mean absolute error: {self.mae(pred, y):.2f}')
        print(f'Root mean squared error: {self.rmse(pred, y):.2f}')

        return pred

    @staticmethod
    def round_up(z):
        return int(math.ceil(z / 10.0)) * 10.0

    @staticmethod
    def equal_array(a, b):
        if b.shape != a.shape:
            return False
        for ai, bi in zip(a.flat, b.flat):
            if ai != bi: return False
        return True

    def make_graphs(self, **kwargs):

        # not working

        y_pred = kwargs.get('y_pred', self.predict_compare())
        y = kwargs.get('y', self.y_test)

        fig = plt.figure()

        fig.ylabel('Actual')
        fig.xlabel('Predicted')

        if np.amax(y) > np.amax(y_pred):
            max = self.round_up(np.amax(y)) + 10
        else:
            max = self.round_up(np.amax(y_pred)) + 10

        tick_list = list()
        tick_spacing = max // 5
        max = int(max)
        for i in range(max):
            if i % tick_spacing:
                tick_list.append(i)

        fig.set_xticks(tick_list)
        fig.set_yticks(tick_list)
        fig.ylim([0, max])
        fig.xlim([0, max])

        fig.scatter(y_pred, y)

        fig.plot([0, max], [0, max])

        if self.equal_array(y, self.y_test):
            fig.set_title('Comparing actual values to predicted values\
                          for test set')
        elif self.equal_array(y, self.y_train):
            fig.set_title('Comparing actual values to predicted values\
                          for training set')
        else:
            fig.set_title('Comparing actual values to predicted values')

        plt.show()


data, target = boston_data()

test = LinearRegression(data, target)
test.make_graphs()
test.fit()
test.make_graphs()
