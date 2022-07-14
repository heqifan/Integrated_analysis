# -*- coding: utf-8 -*-
"""
Created on Wed Jul  6 19:39:01 2022

@author: HYF
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
#导入数据
diabetes_X, diabetes_y = datasets.load_diabetes(return_X_y=True)
diabetes_X = diabetes_X[:, np.newaxis, 2]
#划分测试集验证集
diabetes_X_train = diabetes_X[:-20]
diabetes_X_test = diabetes_X[-20:]
diabetes_y_train = diabetes_y[:-20]
diabetes_y_test = diabetes_y[-20:]
# 创建线性回归模型
regr = linear_model.LinearRegression()
# 训练模型
regr.fit(diabetes_X_train, diabetes_y_train)
# 预测
diabetes_y_pred = regr.predict(diabetes_X_test)
# 模型评价
print('r2_score: %.2f'
      % r2_score(diabetes_y_test, diabetes_y_pred))
# 绘制预测效果图
plt.scatter(diabetes_X_test, diabetes_y_test,  color='black')
plt.plot(diabetes_X_test, diabetes_y_pred, color='blue', linewidth=3)
plt.xticks(())
plt.yticks(())
plt.show()