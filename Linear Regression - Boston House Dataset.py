import numpy as np
from numpy import ndarray
from typing import Callable, Dict, Tuple, List
from sklearn.datasets import load_boston
from sklearn.preprocessing import StandardScaler
import time

start_time = time.time()


boston = load_boston()
data = boston.data
target = boston.target
features = boston.feature_names

s = StandardScaler()
data = s.fit_transform(data)
default_batch_size = round(0.1 * len(data))
group = Tuple[ndarray, ndarray, ndarray, ndarray]

print('Started')

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

    X_test = X[0: round(len(X) * split), :]
    y_test = y[0: round(len(X) * split), :]

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
    print(f'Pred= {Pred[0:5]}')
    print(f'y= {y[0:5]}')

    return


def train(X, y, weights: Dict[str, ndarray], learning_rate: float = 0.001, epochs: int = 10000):

    for i in range(epochs):
        X_batch, y_batch = random_batch(X, y, round(0.1 * len(X)))
        forward = move_forward(X_batch, y_batch, weights)
        lossgrads = loss_grads(forward, weights)

        for key in weights.keys():
            weights[key] -= learning_rate * lossgrads[key]

    return weights


X_train, y_train, X_test, y_test = random_train_test_split(data, target)
weights = create_weights(X_train)

predict_compare(X_test, y_test, weights)
train(X_train,  y_train, weights)
predict_compare(X_test, y_test, weights)

predict_compare(X_train, y_train, weights)


# TODO: Create a plot of error over training
# Turn this into class
# to do sort out train batching
# print results to file
# create time checkmark to print to file


finish_time = time.time()

print(f'Time taken = {finish_time - start_time:.2f}s')
