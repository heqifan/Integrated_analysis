# -- coding: utf-8 --
# from sklearn.datasets import make_classification
# from sklearn.ensemble import GradientBoostingClassifier
# from sklearn.model_selection import cross_val_score
# from bayes_opt import BayesianOptimization
# x, y = make_classification(n_samples=1000,n_features=10,n_classes=2)
# gbdt = GradientBoostingClassifier()
# cross_val_score(gbdt, x, y, cv=20, scoring='roc_auc').mean()
# from bayes_opt import BayesianOptimization
#
# def gbdt_cv(n_estimators, min_samples_split, max_features, max_depth):
#     res = cross_val_score(
#         GradientBoostingClassifier(n_estimators=int(n_estimators),
#                                                         min_samples_split=int(min_samples_split),
#                                                         max_features=min(max_features, 0.999), # float
#                                                         max_depth=int(max_depth),
#                                                         random_state=2
#         ),
#         x, y, scoring='roc_auc', cv=5
#     ).mean()
#     return res
#
# gbdt_op = BayesianOptimization(
#         gbdt_cv,
#         {'n_estimators': (10, 250),
#         'min_samples_split': (2, 25),
#         'max_features': (0.1, 0.999),
#         'max_depth': (5, 15)}
#     )
#
# # print(gbdt_op.maximize())
# print(gbdt_op.max)

# 导入包
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt
import catboost as cb
from bayes_opt import BayesianOptimization


class optimizationClass():
    # 初始化的时候把数据集传进来
    def __init__(self, df):
        self.df = df

        # 定义一个评估指标的函数评估模型好坏

    def reg_calculate(self, true, prediction):
        mse = metrics.mean_squared_error(true, prediction)
        rmse = np.sqrt(mse)
        mae = metrics.mean_absolute_error(true, prediction)
        mape = np.mean(np.abs((true - prediction) / true)) * 100
        r2 = metrics.r2_score(true, prediction)
        rmsle = np.sqrt(metrics.mean_squared_log_error(true, prediction))
        # print("mse: {}, rmse: {}, mae: {}, mape: {}, r2: {}, rmsle: {}".format(mse, rmse, mae, mape, r2, rmsle))
        # return mse, rmse, mae, mape, r2, rmsle
        return {"mse": mse, "rmse": rmse, "mae": mae, "mape": mape, "r2": r2, "rmsle": rmsle}

        # 优化函数 参数为你需要调参的 hyper Parameter 这里用catBoost作为样例

    def optimization_function(self, iterations, learning_rate, depth, l2_leaf_reg):
        parameterDict = {"iterations": int(iterations), "learning_rate": float(learning_rate), "depth": int(depth),
                         "l2_leaf_reg": float(l2_leaf_reg),
                         "task_type": "CPU", "logging_level": "Silent"}
        CB_Regressor = cb.CatBoostRegressor(**parameterDict)
        CB_Regressor.fit(self.df["X_train"], self.df["Y_train"])
        Y_pre = CB_Regressor.predict(self.df["X_test"])
        resDict = self.reg_calculate(self.df["Y_test"], Y_pre)
        return resDict["r2"]

    # 定义一下模型
    def run(self, init_points=2, n_iter=3):
        cb_bo = BayesianOptimization(
            self.optimization_function,
            {'iterations': (200, 5000),
             'learning_rate': (1e-6, 1e-2),
             'depth': (2, 15),
             'l2_leaf_reg': (0, 5)}
        )
        cb_bo.maximize(
            init_points=init_points,
            n_iter=n_iter)
        print("Final result:", cb_bo.max)

from sklearn import datasets  # 导入库

boston = datasets.load_boston()  # 导入波士顿房价数据
print(boston.keys())  # 查看键(属性)     ['data','target','feature_names','DESCR', 'filename']
print(boston.data.shape,boston.target.shape)  # 查看数据的形状 (506, 13) (506,)
print(boston.feature_names)  # 查看有哪些特征 这里共13种
print(boston.DESCR)  # described 描述这个数据集的信息
print(boston.filename)  # 文件路径


from sklearn.model_selection import train_test_split
# check data shape
print("boston.data.shape %s , boston.target.shape %s"%(boston.data.shape,boston.target.shape))
train = boston.data  # sample
target = boston.target  # target
# 切割数据样本集合测试集
X_train, X_test, Y_train, Y_test = train_test_split(train, target, test_size=0.2)  # 20%测试集；80%训练集


op = optimizationClass({"X_train":X_train,"X_test":X_test,"Y_train":Y_train,"Y_test":Y_test})
op.run(init_points=20,n_iter=30)

#


