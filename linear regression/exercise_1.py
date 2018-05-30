# -!- coding=utf-8 -!-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def compute_cost(X, y, theta):
    """代价函数"""
    inner = np.power((X*theta.T-y), 2)
    return np.sum(inner)/(2*len(X))


def gradient_descent(X, y, theta, alpha, iters):
    """
    批量梯度下降
    :param X:
    :param y:
    :param theta:
    :param alpha:学习率
    :param iters: 迭代次数
    :return: theta cost:返回最优theta 和对应cost
    """
    temp = np.matrix(np.zeros(theta.shape))
    parameters = int(theta.ravel().shape[1])    # 列数
    cost = np.zeros(iters)  # 保存每次迭代的cost值

    for i in range(iters):
        error = (X*theta.T) - y

        for j in range(parameters):
            term = np.multiply(error, X[:, j])
            temp[0, j] = theta[0, j]-((alpha/len(X))*np.sum(term))

        theta = temp
        cost[i] = compute_cost(X, y, theta)

    return theta, cost


path = 'ex1data1.txt'
data = pd.read_csv(path, header=None, names=['Population', 'Profit'])
data.insert(0, 'Ones', 1)
cols = data.shape[1]
# X为data所有行，去除最后一列
X = data.iloc[:, 0:cols-1]
# y为data所有行,最后一列
y = data.iloc[:, cols-1:cols]

# 将X和y转化成矩阵
X = np.matrix(X.values)
y = np.matrix(y.values)
# theta初始化为[0,0]
theta = np.matrix(np.array([0, 0]))

# print(compute_cost(X, y, theta))

# 定义学习率和迭代次数
alpha = 0.01
iters = 1000
g, cost = gradient_descent(X, y, theta, alpha, iters)
# compute_cost(X, y, g)


# x = np.linspace(data.Population.min(), data.Population.max(), 100)
# f = g[0, 0] + (g[0, 1] * x)  # f = theta_0+theta_1*x
# fig, ax = plt.subplots(figsize=(12, 8))
# ax.plot(x, f, 'r', label='Prediction')
# ax.scatter(data.Population, data.Profit, label='Training Data')
# ax.legend(loc=2)    # 图例摆放位置
# ax.set_xlabel("Population")
# ax.set_ylabel("Profit")
# ax.set_title("Predicted Profit vs. Population Size")
# plt.show()


#  迭代过程中cost变化
# fig, ax = plt.subplots(figsize=(12, 8))
# ax.plot(np.arange(iters), cost, 'r')
# ax.set_xlabel('Iteration')
# ax.set_ylabel('Cost')
# ax.set_title('Error vs. Training Epoch')
# plt.show()