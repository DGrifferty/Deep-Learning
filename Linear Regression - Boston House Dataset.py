import numpy as np
from numpy import ndarray
from typing import Callable, Dict, Tuple, List
from sklearn.datasets import load_boston
from sklearn.preprocessing import StandardScaler

boston = load_boston()
data = boston.data
target = boston.target
features = boston.feature_names

s = StandardScaler()
data = s.fit_transform(data)


def create_weights(X: ndarray, bias: int =1) -> Dict[str, ndarray]:
    weights = dict()
    weights['W'] = np.random.rand(X.shape[1])
    weights['B'] = np.full((1, 1), 1)
    return weights


def random_train_test_split(X: ndarray, y: ndarray, split: float = 0.75) -> Tuple[ndarray, ndarray, ndarray, ndarray]:

    assert X.ndim == 2
    assert y.ndim == 1
    assert 0 < split < 1

    r = np.c_[X.reshape(len(X), -1), y.reshape(len(y), -1)]
    X = r[:, :X.size // len(X)].reshape(X.shape)
    y = r[:, X.size // len(X):].reshape(y.shape)

    np.random.shuffle(r)

    X_train = X[0: round(len(X) * split), :]
    y_train = y[0: round(len(X) * split)]

    X_test = X[round(len(X) * split):, :]
    y_test = y[round(len(X) * split):]


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


def train():
    pass


def mae(y_pred: ndarray, y: ndarray) -> float:
    return np.mean(np.abs(y - y_pred))


def rmse(y_pred: ndarray, y: ndarray) -> float:
    return np.sqrt(np.mean(np.power(y - y_pred, 2)))


X_train, y_train, X_test, y_test = random_train_test_split(data, target)
weights = create_weights(X_train)
loss, forward = predict(X_train, y_train, weights)

    
    













