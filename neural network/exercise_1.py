# -!- coding=utf-8 -!-

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import matplotlib
from sklearn.metrics import classification_report


def load_data(path, transpose=True):
    """加载数据"""
    data = sio.loadmat(path)
    y = data.get('y')
    y = y.reshape(y.shape[0])
    X = data.get('X')

    if transpose:
        X = np.array([im.reshape((20, 20)) for im in X])
        X = np.array([im.reshape(400) for im in X])

    return X, y


def plot_an_image(image):
    """显示一张图片"""
    fig, ax = plt.subplots(figsize=(2, 2))
    ax.matshow(image.reshape((20, 20)), cmap=matplotlib.cm.binary)
    plt.xticks(np.array([]))
    plt.yticks(np.array([]))


def plot_100_images(X):
    """显示100张图片"""
    size = int(np.sqrt(X.shape[1]))

    sample_idx = np.random.choice(np.arange(X.shape[0]), 100)
    sample_images = X[sample_idx, :]

    fig, ax_array = plt.subplots(nrows=10, ncols=10, sharey=True, sharex=True, figsize=(8, 8))

    for r in range(10):
        for c in range(10):
            ax_array[r, c].matshow(sample_images[10 * r + c].reshape((size, size)), cmap=matplotlib.cm.binary)
            plt.xticks(np.array([]))
            plt.yticks(np.array([]))



def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def load_weight(path):
    data = sio.loadmat(path)
    return data['Theta1'], data['Theta2']


def cost(theta, X, y):
    return np.mean(-y * np.log(sigmoid(X * theta.T)) - (1 - y) * np.log(1 - sigmoid(X * theta.T)))


# 加载数据
X, y = load_data('ex3data1.mat', transpose=False)   # X:(5000, 400) y:(5000, 1)
theta1, theta2 = load_weight('ex3weights')  # theta1:(25, 401) theta2:(10, 26)
# print(X.shape, y.shape)

# # 显示任意一张图片
# pick_one = np.random.randint(0, 5000)
# plot_an_image(X[pick_one, :])
# plt.show()

# # 显示100张图片
# plot_100_images(X)
# plt.show()

X = np.insert(X, 0, values=np.ones(X.shape[0]), axis=1)


# 构建网络
# 第一层网络
a1 = X  # (5000, 401)
# 第二层网络
z2 = a1 @ theta1.T  # (5000, 401) @ (25, 401).T = (5000, 25)
z2 = np.insert(z2, 0, values=np.ones(z2.shape[0]), axis=1)  # 插入bias 项
a2 = sigmoid(z2)
# 第三层网络
z3 = a2 @ theta2.T  # (5000, 26) @ (10, 26).T = (5000, 10)
a3 = sigmoid(z3)
y_pred = np.argmax(a3, axis=1)+1    # 预测结果 (5000, 1)


# 结果统计
print(classification_report(y, y_pred))

