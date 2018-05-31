# -!- coding=utf-8 -!-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as opt

from util import cost
from util import gradient
from util import predict

path = 'ex2data1.txt'
data = pd.read_csv(path, header=None, names=['Exam 1', 'Exam 2', 'Admitted'])

# positive = data[data['Admitted'].isin([1])]
# negative = data[data['Admitted'].isin([0])]

# 绘图
# fig, ax = plt.subplots(figsize=(12, 8))
# ax.scatter(positive['Exam 1'], positive['Exam 2'], s=50, c='r', label='Admitted')
# ax.scatter(negative['Exam 1'], negative['Exam 2'], s=50, c='b', label='Mot Admitted')
# ax.legend()
# ax.set_xlabel('Exam 1 Score')
# ax.set_ylabel('Exam 2 Score')
# plt.show()

data.insert(0,  'Ones', 1)
cols = data.shape[1]
X = data.iloc[:, 0:cols-1]
y = data.iloc[:, cols-1:cols]

# 转化为矩阵
X = np.matrix(X.values)
y = np.matrix(y.values)
# theta 初始化为[0, 0, 0]
theta = np.zeros(3)

# 求出最优theta
result = opt.fmin_tnc(func=cost, x0=theta, fprime=gradient, args=(X, y))
theta_min = np.matrix(result[0])
predictions = predict(theta_min, X)
correct = [1 if (a == 1 and b == 1) or (a == 0 and b ==0) else 0 for (a, b) in zip(predictions, y)]
accuracy = (sum(map(int, correct)) % len(correct))
print('accuracy = {0}%'.format(accuracy))





