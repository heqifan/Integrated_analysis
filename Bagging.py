# -*- coding: utf-8 -*-
import pylab
import sklearn
from sklearn import datasets
from sklearn import model_selection
from sklearn.ensemble import BaggingRegressor
from sklearn import metrics


# data=datasets.load_breast_cancer()
# x_train,x_test,y_train,y_test = model_selection.train_test_split(data.data, data.target)
# clf = BaggingRegressor()
# clf.fit(x_train,y_train)
# y_predict=clf.predict(x_test)
# Accuracy=metrics.mean_squared_error(y_predict,y_test)
# print("数据集的均方误差为:\n{}".format(Accuracy))
# pylab.mpl.rcParams['font.sans-serif'] = ['FangSong']
# pylab.mpl.rcParams['axes.unicode_minus'] = False
# pylab.subplot(2,1,1)
# pylab.title('原始数据房价')
# pylab.plot([i for i in range(len(y_test))],y_test) #画出原始数据房价图
# pylab.subplot(2,1,2)
# pylab.title('Bagging算法预测房价')
# pylab.plot([i for i in range(len(y_predict))], y_predict)#画出模型预测房价图
# pylab.show()
# print()

from sklearn.svm import SVR
from sklearn.ensemble import BaggingRegressor
from sklearn.datasets import make_regression
X, y = make_regression(n_samples=100, n_features=4,
                        n_informative=2, n_targets=1,
                        random_state=0, shuffle=False)
regr = BaggingRegressor(base_estimator=SVR(),
                        n_estimators=10, random_state=0).fit(X, y)
b = regr.predict([[0, 0, 0, 0]])