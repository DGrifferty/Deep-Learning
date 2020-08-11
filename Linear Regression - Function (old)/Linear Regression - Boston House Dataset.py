import numpy as np
from numpy import ndarray
from typing import Callable, Dict, Tuple, List
from sklearn.datasets import load_boston
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt
import time
import math


start_time = time.time()

boston = load_boston()
data = boston.data
target = boston.target
features = boston.feature_names

s = StandardScaler()
data = s.fit_transform(data)
group = Tuple[ndarray, ndarray, ndarray, ndarray]

print('Started')

learning_rate_default = 0.0001
epochs_default = 4000


def create_weights(X: ndarray) -> Dict[str, ndarray]:

    weights = dict()
    weights['W'] = np.random.rand(X.shape[1], 1)
    weights['B'] = np.random.randn(1, 1)

    return weights


def random_train_test_split(X: ndarray, y: ndarray, split: float = 0.75) -> group:

    assert X.ndim == 2
    assert y.ndim == 1
    assert 0 < split < 1

    r = np.c_[X.reshape(len(X), -1), y.reshape(len(y), -1)]
    X = r[:, :X.size // len(X)].reshape(X.shape)
    y = r[:, X.size // len(X):].reshape(-1, 1)

    np.random.shuffle(r)

    X_train = X[0: round(len(X) * split), :]
    y_train = y[0: round(len(X) * split), :]

    X_test = X[round(len(X) * split):, :]
    y_test = y[round(len(X) * split):, :]

    return X_train, y_train, X_test, y_test


def random_batch(X: ndarray, y: ndarray, batch_size: int) -> Tuple[ndarray, ndarray]:
    assert X.ndim == y.ndim == 2

    r = np.c_[X.reshape(len(X), -1), y.reshape(len(y), -1)]
    X = r[:, :X.size // len(X)].reshape(X.shape)
    y = r[:, X.size // len(X):].reshape(-1, 1)

    np.random.shuffle(r)

    X_batch = X[0: batch_size, :]
    y_batch = y[0: batch_size, :]

    return X_batch, y_batch


def move_forward(X_batch: ndarray, y_batch: ndarray, weights: Dict[str, ndarray]) -> Tuple[float, Dict[str, ndarray]]:

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


def predict_compare(X, y, weights):

    N = np.dot(X, weights['W'])
    Pred = N + weights['B']

    print(f'Mean absolute error: {mae(Pred, y):.2f}')
    print(f'Root mean squared error: {rmse(Pred, y):.2f}')

    return Pred


def train(X, y, weights: Dict[str, ndarray], learning_rate: float = learning_rate_default, epochs: int = epochs_default):

    for i in range(epochs):
        X_batch, y_batch = random_batch(X, y, round(0.1 * len(X)))
        forward = move_forward(X_batch, y_batch, weights)
        lossgrads = loss_grads(forward, weights)

        for key in weights.keys():
            weights[key] -= learning_rate * lossgrads[key]

    return weights


def print_results_to_file(y_train, pred_train, y_test, pred_test, starting_weights, starting_bias, weights, time_taken,
                          learning_rate: float = learning_rate_default, epochs: int = epochs_default):
    with open('Linear Regression - Boston House Dataset - Results.txt', 'w+') as f:
        np.set_printoptions(precision=2)
        f.write(f'Learning rate - {learning_rate} \n')
        f.write(f'Epochs - {epochs} \n')
        f.write(f'Time taken - {time_taken:.2f} \n')
        f.write(f'Starting weights - {starting_weights} \n')
        f.write(f'Final weights - {weights["W"]} \n')
        f.write(f'Starting Bias - {starting_bias} \n')
        f.write(f'Final - Bias - {weights["B"]} \n')
        f.write(f'Mean absolute error training set: {mae(pred_train, y_train):.2f} \n')
        f.write(f'Root mean squared error training set: {rmse(pred_train, y_train):.2f} \n')
        f.write(f'Mean absolute error test set: {mae(pred_test, y_test):.2f} \n')
        f.write(f'Root mean squared error test set: {rmse(pred_test, y_test):.2f} \n')

        difference_train = np.array(list())

        for index, value in enumerate(y_train):

            difference_train = np.concatenate((difference_train, pred_train[index] - y_train[index]))
            f.write(f'y_train[{index}] = {value}, pred_train[{index}] = {pred_train[index]} \
             difference = {difference_train[index]:.2f} \n')

        f.write(f'Total difference training set-  {difference_train.sum():.2f}\n ')
        f.write(f'Mean difference training set- {difference_train.mean():.2f}\n ')

        difference_test = np.array(list())

        for index, value in enumerate(y_test):
            difference_test = np.concatenate((difference_test, pred_test[index] - y_test[index]))
            f.write(f'y_test[{index}] = {value}, pred_test[{index}] = {pred_test[index]}\
             difference = {difference_test[index]:.2f} \n')

        f.write(f'Total difference test set-  {difference_test.sum():.2f}\n ')
        f.write(f'Mean difference test set- {difference_test.mean():.2f}\n ')


def round_up(z):
    return int(math.ceil(z / 10.0)) * 10.0


def make_graphs(y, y_pred):
    plt.ylabel('Actual')
    plt.xlabel('Predicted')

    if np.amax(y) > np.amax(y_pred):
        max = round_up(np.amax(y)) + 10
    else:
        max = round_up(np.amax(y_pred)) + 10

    plt.ylim([0, max])
    plt.xlim([0, max])

    plt.scatter(y_pred, y)
    plt.plot([0, max], [0, max])
    plt.show()


X_train, y_train, X_test, y_test = random_train_test_split(data, target)
weights = create_weights(X_train)
starting_weights = weights['W'].copy()
starting_bias = weights['B'].copy()

predict_compare(X_test, y_test, weights)
train(X_train,  y_train, weights)
print('Predict compare - train')
pred_train = predict_compare(X_train, y_train, weights)
print('predict compare - test')
pred_test = predict_compare(X_test, y_test, weights)

# TODO: Create a plot of error over training
# Turn this into class
# to do sort out train batching
# print results to file
# create time check mark to print to file
# Create doc string

time_taken = time.time()-start_time

print(f'Time taken = {time_taken:.2f}s')

print_results_to_file(y_train, pred_train, y_test, pred_test, starting_weights, starting_bias, weights, time_taken)
make_graphs(y_test, pred_test)
