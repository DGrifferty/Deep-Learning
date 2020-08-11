import numpy as np
from numpy import ndarray
from typing import Callable, Dict, Tuple, List
from sklearn.datasets import load_boston
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import time
import math
import operator
import logging

class ML_NN_Methods:

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

        self.X_test = self.X[round(len(self.X) * self.split):, :]
        self.y_test = self.y[round(len(self.X) * self.split):, :]

        self.weights = dict()
        self.weights['W'] = np.random.rand(self.X.shape[1], 1)
        self.weights['B'] = np.random.randn(1, 1)

        self.weights_unchanged = np.copy(self.weights['W'])

        self.learning_rate = 0.001
        self.epochs = 10000
        self.batch_size = 50  # Make variable for different data sizes

        self.__repr__()

    def __repr__(self):
        pass


    def log_time(original_function):
        def wrapper(*args, **kwargs):
            logging.basicConfig(
                level=logging.INFO,
                filename='message.log',
                filemode='w'
            )

            t1 = time.time()

            result = original_function(*args, **kwargs)

            t2 = time.time()

            logging.info(
                f'function {original_function.__name__} ran with kwargs'
                f' {kwargs.keys()} in {t2 - t1:.2f}s')

            return result

        return wrapper

    @log_time
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

    @log_time
    def make_graphs(self, **kwargs):

        y_pred = kwargs.get('y_pred', self.predict_compare())
        y = kwargs.get('y', self.y_test)

        if np.amax(y) > np.amax(y_pred):
            max = self.round_to_10(np.amax(y)) + 10
        else:
            max = self.round_to_10(np.amax(y_pred)) + 10

        tick_list = list()
        tick_spacing = round((0.1 * max) / 5) * 5

        max = int(max)
        for i in range(max + 1):
            if i % tick_spacing == 0:
                tick_list.append(i)

        plt.rcParams['font.size'] = 8
        plt.ylim([0, max])
        plt.xlim([0, max])
        plt.ylabel('Actual')
        plt.xlabel('Predicted')

        plt.scatter(y_pred, y, c='red', marker='^', label='Data')

        plt.xticks(tick_list)
        plt.yticks(tick_list)

        plt.plot([0, max], [0, max], c='black', label='Linear')

        plt.legend(loc='best')

        if self.equal_array(self.weights_unchanged,
                            self.weights['W']):
            if self.equal_array(y, self.y_test):
                plt.title(
                    'Comparing actual values to predicted values'
                    ' before fitting for test set')
            elif self.equal_array(y, self.y_train):
                plt.title(
                    'Comparing actual values to predicted values'
                    ' before fitting for training set')
        elif self.equal_array(y, self.y_test):
            plt.title('Comparing actual values to predicted values'
                      ' for test set')
        elif self.equal_array(y, self.y_train):
            plt.title('Comparing actual values to predicted values'
                      ' for training set')
        else:
            plt.title('Comparing actual values to predicted values')

        plt.show()

    @staticmethod
    def round_to_10(z):
        return int(math.ceil(z / 10.0)) * 10.0

    @staticmethod
    def equal_array(a, b):
        if b.shape != a.shape:
            return False
        for ai, bi in zip(a.flat, b.flat):
            if ai != bi: return False
        return True

    @staticmethod
    def mae(y_pred: ndarray, y: ndarray) -> ndarray:
        return np.mean(np.abs(y - y_pred))

    @staticmethod
    def rmse(y_pred: ndarray, y: ndarray) -> float:
        return np.sqrt(np.mean(np.power(y - y_pred, 2)))


# New methods for NN - >
    def sigmoid(self, array):
        return 1/(1+np.exp(-array))




class LR(ML_NN_Methods):

    @ML_NN_Methods.log_time
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

    @ML_NN_Methods.log_time
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

    @ML_NN_Methods.log_time
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


    @log_time
    def predict_compare(self, **kwargs):

        X = kwargs.get('X', self.X_test)
        y = kwargs.get('y', self.y_test)
        weights = kwargs.get('weights', self.weights)

        pred = np.dot(X, weights['W']) + weights['B']

        if self.equal_array(self.weights_unchanged,
                            self.weights['W']):
            values = False
            if self.equal_array(y, self.y_test):
                print('For test set before fitting:')
            elif self.equal_array(y, self.y_train):
                print('For training set before fitting')
        elif self.equal_array(y, self.y_test):
            print(f'For test set after fitting-')
            values = True
        elif self.equal_array(y, self.y_train):
            print(f'For training set after fitting-')
            values = True

        if values == True:
            print(f'Learning rate - {self.learning_rate}\n'
                  f'Epochs - {self.epochs}\n'
                  f'Batch size - {self.batch_size}')

        print(f'Mean absolute error: {self.mae(pred, y):.2f}')
        print(f'Root mean squared error: {self.rmse(pred, y):.2f}')
        print(f'Mean of y - {np.mean(y)}')
        print(f'Mean of y_pred {np.mean(pred)}')
        print(f'Results are off by an average of:  '
              f'{(self.rmse(pred, y)/np.mean(y))*100} %')
        print('--------------------------------------')

        return pred

class nn(ML_NN_Methods):

    # def __init__(self, data, target):
    #     super().__init__(self, data, target)
    #     print(self.weights)

    def __init__(self, X, y, split = 0.25):

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

        self.X_test = self.X[round(len(self.X) * self.split):, :]
        self.y_test = self.y[round(len(self.X) * self.split):, :]

        self.weights = dict()
        self.weights['W1'] = np.random.rand(self.X.shape[1], 13)
        self.weights['B1'] = np.random.randn(1, 13)

        self.weights['W2'] = np.random.rand(13, 1)
        self.weights['B2'] = np.random.randn(1, 1)

        self.weights_unchanged = np.copy(self.weights['W1'])

        self.learning_rate = 0.001
        self.epochs = 10000
        self.batch_size = 50  # Make variable for different data sizes

        self.__repr__()




    @ML_NN_Methods.log_time
    def move_forward(self, **kwargs):
        X_batch = kwargs.get('X_batch', self.X_train)
        y_batch = kwargs.get('y_batch', self.y_train)
        weights = kwargs.get('weights', self.weights)

        assert X_batch.shape[0] == y_batch.shape[0]
        assert X_batch.shape[1] == weights['W'].shape[0]
        assert weights['B'].shape[0] == weights['B'].shape[1] == 1

        M1 = np.dot(X, weights['W1'])
        N1 = M1 + weights['B1']

        O1 = sigmoid(N1)

        M2 = np.dot(O1, weights['W2'])

        P = M2 +weights['B2']

        forward = dict()
        forward['X'] = X
        forward['M1'] = M1
        forward['N1'] = N1
        forward['O1'] = O1
        forward['M2'] = M2
        forward['P'] = P
        forward['y'] = y

        self.forward = forward

        return forward


    @ML_NN_Methods.log_time
    def loss_grads(self, **kwargs):
        forward = kwargs.get('forward', self.forward)
        weights = kwargs.get('weights', self.weights)

        dLdP = -(forward['y'] - forward_['P'])
        dPdM2 = np.ones_like(forward_info['M2'])
        dLdM2 = dLdP * dPdM2
        dPdB2 = np.ones_like(weights['B2'])
        dLdB2 = (dLdP * dPdB2).sum(axis=0)
        dM2dW2 = np.transpose(forward['O1'], (1, 0))
        dLdW2 = np.dot(dM2dW2, dLdP)
        dM2dO1 = np.transpose(weights['W2'], (1, 0))
        dLdO1 = np.dot(dLdM2, dM2dO1)
        dO1dN1 = sigmoid(forward_info['N1']) * (
                    1 - sigmoid(forward['N1']))

        dLdN1 = dLdO1 * dO1dN1
        dN1dB1 = np.ones_like(weights['B1'])
        dN1dM1 = np.ones_like(forward['M1'])
        dLdB1 = (dLdN1 * dN1dB1).sum(axis=0)
        dLdM1 = dLdN1 * dN1dM1
        dM1dW1 = np.transpose(forward['X'], (1, 0))
        dLdW1 = np.dot(dM1dW1, dLdM1)

        lossgrad: Dict[str, ndarray] = {}
        lossgrad['W2'] = dLdW2
        lossgrad['B2'] = dLdB2.sum(axis=0)
        lossgrad['W1'] = dLdW1
        lossgrad['B1'] = dLdB1.sum(axis=0)

        self.lossgrad = lossgrad

        return lossgrad

    def predict(self, ):

    @ML_NN_Methods.log_time
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

    @log_time
    def predict_compare(self, **kwargs):
        pass

        # X = kwargs.get('X', self.X_test)
        # y = kwargs.get('y', self.y_test)
        # weights = kwargs.get('weights', self.weights)
        #
        # pred = np.dot(X, weights['W']) + weights['B']
        #
        # if self.equal_array(self.weights_unchanged,
        #                     self.weights['W']):
        #     values = False
        #     if self.equal_array(y, self.y_test):
        #         print('For test set before fitting:')
        #     elif self.equal_array(y, self.y_train):
        #         print('For training set before fitting')
        # elif self.equal_array(y, self.y_test):
        #     print(f'For test set after fitting-')
        #     values = True
        # elif self.equal_array(y, self.y_train):
        #     print(f'For training set after fitting-')
        #     values = True
        #
        # if values == True:
        #     print(f'Learning rate - {self.learning_rate}\n'
        #           f'Epochs - {self.epochs}\n'
        #           f'Batch size - {self.batch_size}')
        #
        # print(f'Mean absolute error: {self.mae(pred, y):.2f}')
        # print(f'Root mean squared error: {self.rmse(pred, y):.2f}')
        # print(f'Mean of y - {np.mean(y)}')
        # print(f'Mean of y_pred {np.mean(pred)}')
        # print(f'Results are off by an average of:  '
        #       f'{(self.rmse(pred, y)/np.mean(y))*100} %')
        # print('--------------------------------------')
        #
        # return pred



def boston_data() -> Tuple[ndarray, ndarray]:
    boston = load_boston()
    data = boston.data
    target = boston.target
    s = StandardScaler()
    data = s.fit_transform(data)  # Mean zero sd - 1

    return data, target


if __name__ == '__main__':
    data, target = boston_data()

    test = nn(data, target)
    # test.make_graphs()
    # # Should only have to feed in y values to graphs, it should
    # # Call predict_compare - without text itself, if no
    # # y_pred is passed in
    # y_pred = test.predict_compare(X=test.X_train, y=test.y_train,
    #                               weights=test.weights)
    # test.make_graphs(y_pred=y_pred, y=test.y_train)
    # test.fit()
    # test.make_graphs()
    # y_pred = test.predict_compare(X=test.X_train, y=test.y_train,
    #                               weights=test.weights)
    # test.make_graphs(y_pred=y_pred, y=test.y_train)
