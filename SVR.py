# -- coding: utf-8 --
# 从 sklearn.datasets 导入波士顿房价数据读取器。
from sklearn.datasets import load_boston
# 从读取房价数据存储在变量 boston 中。
boston = load_boston()

# 从sklearn.cross_validation 导入数据分割器。
from sklearn.model_selection import train_test_split
X = boston.data
y = boston.target
# 随机采样 25% 的数据构建测试样本，其余作为训练样本。
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=33, test_size=0.25)

# 从 sklearn.svm 中导入支持向量机（回归）模型。
from sklearn.svm import SVR
# 使用线性核函数配置的支持向量机进行回归训练，并且对测试样本进行预测。
linear_svr = SVR(kernel='linear')
linear_svr.fit(X_train, y_train)
linear_svr_y_predict = linear_svr.predict(X_test)

# 使用多项式核函数配置的支持向量机进行回归训练，并且对测试样本进行预测。
poly_svr = SVR(kernel='poly')
poly_svr.fit(X_train, y_train)
poly_svr_y_predict = poly_svr.predict(X_test)

# 使用径向基核函数配置的支持向量机进行回归训练，并且对测试样本进行预测。
rbf_svr = SVR(kernel='rbf')
rbf_svr.fit(X_train, y_train)
rbf_svr_y_predict = rbf_svr.predict(X_test)

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
print('R-squared value of linear SVR is', linear_svr.score(X_test, y_test))
print('The mean squared error of linear SVR is', mean_squared_error(
    ss_y.inverse_transform(y_test), ss_y.inverse_transform(linear_svr_y_predict)))
print('The mean absoluate error of linear SVR is', mean_absolute_error(
    ss_y.inverse_transform(y_test), ss_y.inverse_transform(linear_svr_y_predict)))

print('R-squared value of Poly SVR is', poly_svr.score(X_test, y_test))
print('The mean squared error of Poly SVR is', mean_squared_error(
    ss_y.inverse_transform(y_test), ss_y.inverse_transform(poly_svr_y_predict)))
print('The mean absoluate error of Poly SVR is', mean_absolute_error(
    ss_y.inverse_transform(y_test), ss_y.inverse_transform(poly_svr_y_predict)))

print('R-squared value of RBF SVR is', rbf_svr.score(X_test, y_test))
print('The mean squared error of RBF SVR is', mean_squared_error(
    ss_y.inverse_transform(y_test), ss_y.inverse_transform(rbf_svr_y_predict)))
print('The mean absoluate error of RBF SVR is', mean_absolute_error(
    ss_y.inverse_transform(y_test), ss_y.inverse_transform(rbf_svr_y_predict)))


