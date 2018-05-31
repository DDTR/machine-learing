# -!- coding=utf-8 -!-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt

from util import cost_reg
from util import gradient_reg
from util import predict


path = 'ex2data2.txt'
data = pd.read_csv(path, header=None, names=['Test 1', 'Test 2', 'Accepted'])


# positive = data[data['Accepted'].isin([1])]
# negative = data[data['Accepted'].isin([0])]
#
# fig, ax = plt.subplots(figsize=(12, 8))
# ax.scatter(positive['Test 1'], positive['Test 2'], s=50, c='b', marker='o', label='Accepted')
# ax.scatter(negative['Test 1'], negative['Test 2'], s=58, c='r', marker='x', label='Rejected')
# ax.legend()
# ax.set_xlabel('Test 1 Score')
# ax.set_ylabel('Test 2 Score')
# plt.show()

degree = 5
x1 = data['Test 1']
x2 = data['Test 2']

data.insert(3, 'Ones', 1)
# 创建新的特征值
for i in range(1, degree):
    for j in range(0, i):
        data['F' + str(i) +str(j)] = np.power(x1, i-j) * np.power(x2, j)

# 删除初始特征数据
data.drop('Test 1', axis=1, inplace=True)
data.drop('Test 2', axis=1, inplace=True)

cols = data.shape[1]
X = data.iloc[:, 1:cols]
y = data.iloc[:, 0:1]

X = np.matrix(X.values)
y = np.matrix(y.values)
theta = np.zeros(11)

# 学习率
learing_rate = 1

# 参数优化
result = opt.fmin_tnc(func=cost_reg, x0=theta, fprime=gradient_reg, args=(X, y, learing_rate))
theta_min = np.matrix(result[0])
predictions = predict(theta_min, X)
correct = [1 if (a == 1 and b == 1) or (a == 0 and b == 0) else 0 for a, b in zip(predictions, y)]
accuracy = (sum(map(int, correct)) % len(correct))
print('accuracy = {0}% '.format(accuracy))
