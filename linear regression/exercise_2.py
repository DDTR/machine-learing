# -!- coding=utf-8 -!-
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from exercise_1 import compute_cost
from exercise_1 import gradient_descent

path = 'ex1data2.txt'
data = pd.read_csv(path, header=None, names=['Size', 'Bedroom', 'Price'])

# 归一化
data = (data - data.mean()) / data.std()

# 对数据进行处理
data.insert(0, 'Ones', 1)

cols = data.shape[1]
X = data.iloc[:, 0:cols-1]
y = data.iloc[:, cols-1:cols]

# 转化成矩阵
X = np.matrix(X.values)
y = np.matrix(y.values)
theta = np.matrix(np.array([0, 0, 0]))

# 学习率和迭代次数
alpha = 0.01
iters = 1000

# 梯度下降优化
g, cost = gradient_descent(X, y, theta, alpha, iters)
compute_cost(X, y, theta)


# 迭代过程中cost变化
fig, ax = plt.subplots(figsize=(12, 8))
ax.plot(np.arange(iters), cost, 'r')
ax.set_xlabel('Iterations')
ax.set_ylabel('Cost')
ax.set_title('Error vs. Training Epoch')
plt.show()
