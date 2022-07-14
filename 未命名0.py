# -*- coding: utf-8 -*-
"""
Created on Sat Jul  9 19:06:31 2022

@author: HYF
"""

###########################################
#各种机器学习算法在一个回归数据集上的使用
###########################################
 
###############
#主要机器学习算法
#############################################################
#线性回归/Ridge/Lasso/弹性网
#决策树/RF(随机森林)/GBDT/XGBoost
#LinearSVM/SVM/B-P神经网络
#############################################################
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 
rng = np.random.RandomState(1)
X = np.linspace(0, 6, 100)[:, np.newaxis]
y = np.sin(X).ravel() + np.sin(6 * X).ravel() + rng.normal(0, 0.1, X.shape[0])
plt.scatter(X, y, c="k",label="training samples")