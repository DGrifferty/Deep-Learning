import numpy as np
from matplotlib import pyplot as plt


arry = np.arange(-5, 5, 0.2)
arrx = np.copy(arry)


def sigmoid(array):
    return 1 / (1 + np.exp(-array))

arry = sigmoid(arry)

plt.title('Sigmoid Function')
plt.ylabel('y')
plt.xlabel('x')

plt.plot(arrx, arry, c='red')
plt.show()
