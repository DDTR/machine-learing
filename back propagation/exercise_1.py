# -!- coding=utf-8 -!-
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import matplotlib
import scipy.optimize as opt
from sklearn.metrics import classification_report


def load_data(path, transpose=True):
    """加载数据"""
    data = sio.loadmat(path)
    y = data.get('y')
    y = y.reshape(y.shape[0])
    X = data.get('X')

    if transpose:
        X = np.array([im.reshape((20, 20)).T for im in X])
        X = np.array([im.reshape(400) for im in X])

    return X, y


def load_weight(path):
    """加载权重数据"""
    data = sio.loadmat(path)
    return data['Theta1'], data['Theta2']


def plot_100_images(X):
    size = int(np.sqrt(X.shape[1]))

    sample_idx = np.random.choice(np.arange(X.shape[0]), 100)
    sample_images = X[sample_idx, :]
    fig, ax_array = plt.subplots(nrows=10, ncols=10, sharex=True, sharey=True, figsize=(8, 8))

    for r in range(10):
        for c in range(10):
            ax_array[r, c].matshow(sample_images[10 * r + c].reshape((size, size)), cmap=matplotlib.cm.binary)

            plt.xticks(np.array([]))
            plt.yticks(np.array([]))


def serialize(a, b):
    return np.concatenate((np.ravel(a), np.ravel(b)))


def expand_y(y):
    res = []
    for i in y:
        y_array = np.zeros(10)
        y_array[i - 1] = 1

        res.append(y_array)

    return np.array(res)


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def deserialize(seq):
    return seq[:25 * 401].reshape(25, 401), seq[25 * 401:].reshape(10, 26)


def feed_forward(theta, X):
    """前向传播"""
    t1, t2 = deserialize(theta) # t1:(25,401) t2:(10,26)
    m = X.shape[0]  # 输入样本个数
    a1 = X  # X;(5000, 401)

    z2 = a1 @ t1.T  # z2:(5000, 25)
    a2 = np.insert(sigmoid(z2), 0, np.ones(m), axis=1)  # a2:(5000, 26)

    z3 = a2 @ t2.T  # z3:(5000, 10)
    h = sigmoid(z3) # 输出

    return a1, z2, a2, z3, h


def cost(theta, X, y):
    m = X.shape[0]
    _, _, _, _, h = feed_forward(theta, X)

    pair_computation = np.multiply(y, np.log(h)) - np.multiply((1 - y), np.log(1 - h))  # 计算与实际标签y之间的误差

    return pair_computation.sum() / m


def regularized_cost(theta, X, y, l=1):
    """正则化代价函数"""
    t1, t2 = deserialize(theta) # 解序列是因为要正则化
    m = X.shape[0]

    reg_t1 = (1 / (2 * m)) * np.power(t1[:, 1:], 2).sum()   # 不含bais项
    reg_t2 = (1 / (2 * m)) * np.power(t2[:, 1:], 2).sum()

    return cost(theta, X, y) + reg_t1 + reg_t2


def sigmoid_gradient(z):
    return np.multiply(sigmoid(z), 1 - sigmoid(z))


def gradient(theta, X, y):
    t1, t2 = deserialize(theta) # t1:(25, 401) t2:(10,26)
    m = X.shape[0]  # 输入样本个数

    delta1 = np.zeros(t1.shape) # delta1:(25, 401)
    delta2 = np.zeros(t2.shape) # detal2:(10, 26)

    a1, z2, a2, z3, h = feed_forward(theta, X)  # 前向传递

    for i in range(m):
        a1i = a1[i, :]  # (1, 401)
        z2i = z2[i, :]  # (1, 25)
        a2i = a2[i, :]  # (1. 26)

        hi = h[i, :]    # (1, 10)
        yi = y[i, :]    # (1, 10)

        d3i = hi - yi   # (1, 10)

        z2i = np.insert(z2i, 0, np.zeros(1))    # (1. 26)
        d2i = np.multiply(t2.T @ d3i, sigmoid_gradient(z2i))    # 反向传播计算误差

        delta2 += np.matrix(d3i).T @ np.matrix(a2i)
        delta1 += np.matrix(d2i[1:]).T @ np.matrix(a1i)

    delta1 = delta1 / m
    delta2 = delta2 / m

    return serialize(delta1, delta2)


def regularized_gradient(theta, X, y, l=1):
    m = X.shape[0]
    delta1, delta2 = deserialize(gradient(theta, X, y))
    t1, t2 = deserialize(theta)

    t1[:, 0] = 0
    reg_term_d1 = (1 / m) * t1
    delta1 = delta1 + reg_term_d1

    t2[:, 0] = 0
    reg_term_d2 = (1 / m) * t2
    delta2 = delta2 + reg_term_d2

    return serialize(delta1, delta2)


def expand_array(arr):
    return np.array(np.matrix(np.ones(arr.shape[0])).T @ np.matrix(arr))


def gardient_checking(theta, X, y, epsilon, regularized=False):
    def a_numeric_grad(plus, minus, regularized=False):
        if regularized:
            return (regularized_cost(plus, X, y) - regularized_cost(minus, X, y)) / (epsilon * 2)
        else:
            return (cost(plus, X, y) - cost(minus, X, y)) / (epsilon * 2)

    theta_matrix = expand_array(theta)
    epsilon_matrix = np.identity(len(theta)) * epsilon

    plus_matrix = theta_matrix + epsilon_matrix
    minus_matrix = theta_matrix - epsilon_matrix

    numeric_grad = np.array([a_numeric_grad(plus_matrix[i], minus_matrix[i], regularized)
                            for i in range(len(theta))])

    analytic_grad = regularized_gradient(theta, X, y) if regularized else gradient(theta, X, y)

    diff = np.linalg.norm(numeric_grad - analytic_grad) / np.linalg.norm(numeric_grad +analytic_grad)
    print('If your backpropagation implementation is correct, \nthe relative difference will be smaller than 10e-9 (assume '
          'epsilon=0.0001.\nRelative Diffference:{}\n'.format(diff))


def random_int(size):
    return np.random.uniform(-0.12, 0.12, size)


def nn_training(X, y):
    init_theta = random_int(10285)
    res = opt.minimize(fun=regularized_cost,
                       x0=init_theta,
                       args=(X, y, 1),
                       method='TNC',
                       jac=regularized_gradient,
                       options={'maxiter':400})
    return res


def show_accuracy(theta, X, y):
    _, _, _, _, h = feed_forward(theta, X)

    y_pred = np.argmax(h, axis=1) + 1
    print(classification_report(y_pred, y))


def plot_hidden_layer(theta):
    final_theta1, _ = deserialize(theta)
    hidden_layer = final_theta1[:, 1:]

    fig, ax_array = plt.subplots(nrows=5, ncols=5, sharex=True, sharey=True, figsize=(5, 5))

    for r in range(5):
        for c in range(5):
            ax_array[r, c].matshow(hidden_layer[5 * r + c].reshape((20, 20)),
                                   cmap=matplotlib.cm.binary)
            plt.xticks(np.array([]))
            plt.yticks(np.array([]))


X_raw, y_raw = load_data('ex4data1.mat')
X = np.insert(X_raw, 0, np.ones(X_raw.shape[0]), axis=1)    # 增加一列
y = expand_y(y_raw)
t1, t2 = load_weight('ex4weights.mat')
theta = serialize(t1, t2)   # 将t1和t2序列化

res = nn_training(X, y)
final_theta = res.x
show_accuracy(final_theta, X, y_raw)



# plot_100_images(X_raw)
# plt.show()

