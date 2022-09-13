# -- coding: utf-8 --
# explicitly require this experimental feature
# now you can import normally from model_selection
# from sklearn.datasets import load_iris
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.experimental import enable_halving_search_cv  # noqa
# from sklearn.model_selection import HalvingRandomSearchCV
# from scipy.stats import randint
# import numpy as np
# X, y = load_iris(return_X_y=True)
# clf = RandomForestClassifier(random_state=0)
# np.random.seed(0)
# param_distributions = {"max_depth": [3, None],
#                         "min_samples_split": randint(2, 11)}
# search = HalvingRandomSearchCV(clf, param_distributions,
#                                 resource='n_estimators',
#                                 max_resources=10,
#                                 random_state=0).fit(X, y)
# print(search.best_params_)
# print(search.best_params_["max_depth"])
import lce
print(lce.__version__)