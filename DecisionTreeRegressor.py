# -- coding: utf-8 --
from sklearn.tree import DecisionTreeRegressor
import numpy as np
import matplotlib.pyplot as plt
rng = np.random.RandomState(0)    #随机数种子
X = np.sort(rng.rand(80,1)*5,axis=0)    #创建x
y = np.sin(X).ravel()    #创建y   ravel()函数：降维，此处将二位数组转换为一维
y[::5] += 3*(0.5-rng.rand(16))  #增添噪音
plt.figure()
plt.scatter(X,y)
plt.show()