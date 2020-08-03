import numpy as np
from numpy import ndarray
from typing import Callable, Dict, Tuple, List
from sklearn.datasets import load_boston
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

boston = load_boston()
data = boston.data
target = boston.target
features = boston.feature_names

s = StandardScaler()
data = s.fit_transform(data)

group = Tuple[ndarray, ndarray, ndarray, ndarray]

def create_weights(X: ndarray, bias: int = 1) -> Dict[str, ndarray]:
    weights = dict()
    weights['W'] = np.random.rand(X.shape[1])
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
    y_train = y[0: round(len(X) * split)]

    X_test = X[round(len(X) * split):, :]
    y_test = y[0: round(len(X) * split):]


    return X_train, y_train, X_test, y_test


def predict(X_batch: ndarray, y_batch: ndarray, weights: Dict[str, ndarray]) -> Tuple[float, Dict[str, ndarray]]:

    assert X_batch.shape[0] == y_batch.shape[0]
    assert X_batch.shape[1] == weights['W'].shape[0]
    assert weights['B'].shape[0] == weights['B'].shape[1] == 1

    N = np.dot(X_batch, weights['W'])
    Pred = N + weights['B']

    loss = np.mean(np.power(Pred - y_batch, 2))

    forward = dict()
    forward['X'] = X_batch
    forward['Y'] = y_batch
    forward['N'] = N
    forward['P'] = Pred
    return loss, forward


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


def train(grad: Dict[str, ndarray], weights: Dict[str, ndarray], learning_rate: float = 0.001):
    for key in weights.keys():
        print(key)
        print(weights[key].shape)
        print(grad[key].shape)
        weights[key] -= learning_rate * grad[key]

    return weights


X_train, y_train, X_test, y_test = random_train_test_split(data, target)
weights = create_weights(X_train)
# TODO: Create a plot of error over training
for i in range(100):
    loss, forward = predict(X_train, y_train, weights)
    loss_grads = loss_grads(forward, weights)
    weights = train(loss_grads, weights)


    
    













