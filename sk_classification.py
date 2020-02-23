import json
import math
import pprint
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import *
from sklearn.preprocessing import *
from sklearn.model_selection import train_test_split
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import *
from sklearn.neighbors import *
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.naive_bayes import *
from sklearn.tree import *
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import *
from sklearn.metrics import mean_squared_error,r2_score,f1_score
from sklearn. preprocessing import *

forex = pd.read_csv('prep_forex.csv', header=[0,1], index_col=0)
index = pd.read_csv('prep_index.csv', header=[0,1,2], index_col=0)

forex_pairs = list(set([x[1] for x in forex.columns if x[0] == 'Close']))
index_pairs = list(set([(x[1], x[2]) for x in index.columns if x[0] == 'Close']))

reg_models = [LinearRegression, Ridge, RidgeCV, Lasso, LassoCV, LassoLarsCV, LassoLarsIC, MultiTaskLasso,
              ElasticNet, ElasticNetCV, MultiTaskElasticNet, MultiTaskElasticNetCV, LassoLars,
              OrthogonalMatchingPursuit, BayesianRidge, ARDRegression,
              SGDRegressor, PassiveAggressiveRegressor, HuberRegressor, RANSACRegressor, TheilSenRegressor,
              KernelRidge, SVR, NuSVR, KNeighborsRegressor, RadiusNeighborsRegressor,
              GaussianProcessRegressor, PLSRegression, DecisionTreeRegressor,
              BaggingRegressor, AdaBoostRegressor, ExtraTreesRegressor, GradientBoostingRegressor,
              RandomForestRegressor, HistGradientBoostingRegressor]

cls_models = [RidgeClassifier, LogisticRegression,  LogisticRegression, LogisticRegressionCV,
              SGDClassification, Perception, PassiveAggressiveClassifier, SVC, NuSVC, LinearSVC,
              KNeighborsClassifier, NearestCentroid, GaussianProcessClassifier,
              GaussianNB, MultinomialNB, ComplementNB, BernoulliNB, CategoricalNB,
              DecisionTreeClassifier, BaggingClassifier, AdaBoostClassifier, ExtraTreesClassifier,
              GradientBoostingClassifier, RandomForestClassifier, HistGradientBoostingClassifier]


metric = 'Close'
metrics = ['Open', 'Close', 'Low', 'High']
target = [metric+'_Ret']
n_bins = 3
scalers = [Binarizer, KBinsDiscretizer]
features = ['Intraday_HL', 'Intraday_OC', 'Prev_close_open'] + [y+x for x in ['_Ret', '_Ret_MA_3', '_Ret_MA_15', '_Ret_MA_45', '_MTD', '_YTD'] for y in metrics]

def plot_results(y_true, y_pred, model):
    plot_df = pd.concat([y_true.reset_index(drop=True), pd.DataFrame(y_pred)], axis=1, ignore_index=True)
    plt.figure()
    plt.plot(plot_df)
    plt.title(model().__class__.__name__)

def run_sklearn_model(model, train, test, features, target):
    try:
        X_train, y_train = train
        X_test, y_test = test
    except:
        pass
    reg = model()#max_iter=1000, tol=1e-3)
    reg.fit(X_train, y_train)
    try:
        pprint.pprint(dict(zip(features,reg.feature_importances_)))
    except:
        pass
    y_pred = reg.predict(X_test)
    try:
        return({"MSE":mean_squared_error(y_test, y_pred),
            "R2" :r2_score(y_test, y_pred),
            "F1" :f1_score(y_test, y_pred, average=None),
            "F1 Weighted" : f1_score(y_pred, y_pred, average='weighted')})
    except:
        y_pred = ([x[0] for x in y_pred])
        for i in range(len(y_pred)):
            if(np.isnan(y_pred[i])):
                y_pred[i] = 0
        return({"MSE":mean_squared_error(y_test, y_pred),
            "R2" :r2_score(y_test, y_pred),
            "F1" :f1_score(y_test, y_pred, average=None),
            "F1 Weighted" : f1_score(y_pred, y_pred, average='weighted')})
    # print("MSE: ", mean_squared_error(y_test, y_pred))
    # print("R2: ", r2_score(y_test, y_pred))
    # plot_results(y_test, y_pred, model)

def split_scale(X, y, scaler, shuffle=False):
    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=shuffle, test_size=0.2)
    # X_train = Binarizer().fit_transform(X_train)
    # X_test = Binarizer().fit_transform(X_test)

    if (scaler == scalers[-1]):
        scaler_X = scaler(n_bins=n_bins, encode='ordinal', strategy='uniform')
    else:
        scaler_X = scaler()
    scaler_X = scaler_X.fit(X_train)
    X_train = scaler_X.transform(X_train)
    X_test = scaler_X.transform(X_test)
    if (scaler == scalers[-1]):
        scaler_y = scaler(n_bins=n_bins, encode='ordinal', strategy='uniform')
    else:
        scaler_y = scaler()
    scaler_y = scaler_y.fit(y_train)
    y_train = scaler_y.transform(y_train)
    y_test = scaler_y.transform(y_test)
    return(X_train, X_test, y_train, y_test)


def do_forex(cur, model, transf = None, shuffle=False):
    forex_cols = [x for x in forex.columns if x[1] == cur]
    X = forex[[col for col in forex_cols if col[0] in features + ['Time features']]][:-1]
    y = forex[[col for col in forex_cols if col[0] in target]].shift(-1)[:-1]
    X = X.dropna(how='any')
    y = y[y.index.isin(X.index)]
    X_train, X_test, y_train, y_test = split_scale(X, y, transf, shuffle)
    res = run_sklearn_model(model, (X_train, y_train), (X_test, y_test), features, target)
    return(res)

def do_index(cur, model, transf = None, shuffle=False):
    index_cols = [x for x in index.columns if x[1] == cur[0] and x[2] == cur[1]]
    X = index[[col for col in index_cols if col[0] in features + ['Time features']]][:-1]
    y = index[[col for col in index_cols if col[0] in target]].shift(-1)[:-1]
    X = X.dropna(how='any')
    y = y[y.index.isin(X.index)]
    X_train, X_test, y_train, y_test = split_scale(X, y, transf, shuffle)
    res = run_sklearn_model(model, (X_train, y_train), (X_test, y_test), features, target)
    return(res)

def iterate_markets():
    reg_res = []
    for f_m in (forex_pairs+index_pairs)[:1]:
        print(f_m)
        for model in reg_models[:1]:
            for scaler in scalers:
                for shuffle in [True, False]:
                    try:
                        if(f_m in forex_pairs):
                            res = do_forex(f_m, model, scaler, shuffle)
                        else:
                            res = do_index(f_m, model, scaler, shuffle)
                    except:
                        #res = do_index(f_m, model, scaler, shuffle)
                        continue
                    res['Pair'] = f_m
                    res['Transformation'] = scaler().__class__.__name__ if scaler is not None else None
                    res['Shuffle'] = shuffle
                    res['Model'] = model().__class__.__name__
                    reg_res.append(res)
    return(reg_res)

res = iterate_markets()
res_df = pd.DataFrame(res, columns= ['Pair', 'Model', 'Transformation', 'Shuffle', 'MSE', 'R2'])
# print(res_df)
res_df.to_csv("sk_classification.csv")
# do_stuff(["HKD", "Hang Seng"], LinearRegression)
