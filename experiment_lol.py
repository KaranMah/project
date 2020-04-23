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
from sklearn.gaussian_process import *
from sklearn.cross_decomposition import PLSRegression
from sklearn.naive_bayes import *
from sklearn.tree import *
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import *
from sklearn.metrics import mean_squared_error,r2_score, confusion_matrix, f1_score, plot_confusion_matrix

import warnings
warnings.filterwarnings("ignore")

forex = pd.read_csv('prep_forex.csv', header=[0,1], index_col=0)
index = pd.read_csv('prep_index.csv', header=[0,1,2], index_col=0)

curr = 'MNT'

forex_pairs = list(set([x[1] for x in forex.columns if x[1] == curr]))
index_pairs = list(set([(x[1], x[2]) for x in index.columns if x[1] == curr]))

reg_models = [LinearRegression, Ridge, RidgeCV, Lasso, LassoCV, LassoLarsCV, LassoLarsIC, MultiTaskLasso,
              ElasticNet, ElasticNetCV, MultiTaskElasticNet, MultiTaskElasticNetCV, LassoLars,
              OrthogonalMatchingPursuit, BayesianRidge, ARDRegression,
              SGDRegressor, PassiveAggressiveRegressor, HuberRegressor, RANSACRegressor, TheilSenRegressor,
              KernelRidge, SVR, NuSVR, KNeighborsRegressor, RadiusNeighborsRegressor,
              GaussianProcessRegressor, PLSRegression, DecisionTreeRegressor,
              BaggingRegressor, AdaBoostRegressor, ExtraTreesRegressor, GradientBoostingRegressor,
              RandomForestRegressor, HistGradientBoostingRegressor]

cls_models = [RidgeClassifier, LogisticRegression, LogisticRegressionCV,
              SGDClassifier, Perceptron, PassiveAggressiveClassifier, SVC, NuSVC, LinearSVC,
              KNeighborsClassifier, NearestCentroid, GaussianProcessClassifier,
              GaussianNB, MultinomialNB, ComplementNB, BernoulliNB, CategoricalNB,
              DecisionTreeClassifier, BaggingClassifier, AdaBoostClassifier, ExtraTreesClassifier,
              GradientBoostingClassifier, RandomForestClassifier, HistGradientBoostingClassifier]


scalers = [None, MinMaxScaler, MaxAbsScaler, StandardScaler, RobustScaler,
            PowerTransformer, FunctionTransformer]

metric = 'Close_Ret'
metrics = ['Open', 'Close', 'Low', 'High']
target = [metric]
features = ['Intraday_HL', 'Intraday_OC', 'Prev_close_open'] + [y+x for x in ['_Ret', '_Ret_MA_3', '_Ret_MA_15', '_Ret_MA_45', '_MTD', '_YTD'] for y in metrics]# if (x+y) not in target]


def multiple(x, y = 10):
    return(x*y)

def plot_results(y_true, y_pred, model):
    plot_df = pd.concat([y_true.reset_index(drop=True), pd.DataFrame(y_pred)], axis=1, ignore_index=True)
    plot_df.index = y_true.index
    plot_df.columns = ['y_true', 'y_pred']
    print(plot_df.tail(10))
    plt.figure()
    plt.plot(plot_df)
    plt.legend(['y_true', 'y_pred'])
    plt.title(model().__class__.__name__ + " MNT")
    plt.show()

def run_sklearn_model(model, train, test, features, target):
    try:
        X_train, y_train = train
        X_test, y_test = test
    except:
        pass
    reg = model()#max_iter=1000, tol=1e-3)
    reg.fit(X_train, y_train)
    try:
        pprint.pprint(dict(zip(X_train.columns.values,reg.feature_importances_)))
    except:
        pass
    y_pred = reg.predict(X_test)
    plot_results(pd.DataFrame(y_test), pd.DataFrame(y_pred), model)
    try:
        plot_confusion_matrix(reg, X = X_test, y_true = y_test, display_labels=np.unique(y_test))
        print({"F1 Score" :f1_score(y_test, y_pred)})
    except:
        pass
    try:
        return({"MSE":mean_squared_error(y_test, y_pred),
            "R2" :r2_score(y_test, y_pred)})
    except:
        y_pred = ([x[0] for x in y_pred])
        for i in range(len(y_pred)):
            if(np.isnan(y_pred[i])):
                y_pred[i] = 0
        return({"MSE":mean_squared_error(y_test, y_pred),
            "R2" :r2_score(y_test, y_pred)})
    # print("MSE: ", mean_squared_error(y_test, y_pred))
    # print("R2: ", r2_score(y_test, y_pred))

def bins_scale(X, y, scaler, shuffle=False):
    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=shuffle, test_size=0.2)
    X_train = Binarizer().fit_transform(X_train)
    X_test = Binarizer().fit_transform(X_test)
    y_train = Binarizer().fit_transform(y_train)
    y_test = Binarizer().fit_transform(y_test)
    return(X_train, X_test, y_train, y_test)

def split_scale(X, y, scaler, shuffle=False):
    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=shuffle, test_size=0.2)
    # print(X_train.head())
    # print(y_train.head())
    if(scaler is None):
        print(y_train.head())
        return(X_train, X_test, y_train, y_test)
    else:
        scaler_X = scaler()
        if(scaler == scalers[-1]):
            scaler_X = scaler(np.log1p)
        scaler_X = scaler_X.fit(X_train)
        X_train = scaler_X.transform(X_train)
        X_test = scaler_X.transform(X_test)
        # scaler_y = scaler()
        # if(scaler == scalers[-1]):
        #     scaler_y = scaler(np.log1p)
        # scaler_y = scaler_y.fit(y_train.values.reshape(-1, 1))
        # y_train = scaler_y.transform(y_train.values.reshape(-1,1))
        # y_test = scaler_y.transform(y_test.values.reshape(-1,1))
        X_train = PolynomialFeatures(2).fit_transform(X_train)
        X_test = PolynomialFeatures(2).fit_transform(X_test)
        return(X_train, X_test, y_train, y_test)

def do_forex(cur, model, transf = None, shuffle=False):
    forex_cols = [x for x in forex.columns if x[1] == cur]
    old_y = forex[[col for col in forex_cols if col[0] in target]][:-1]
    X = forex[[col for col in forex_cols if col[0] in features + ['Time features']]][:-1]
    y = forex[[col for col in forex_cols if col[0] in target]].shift(-1)[:-1]
    X = X.dropna(how='any')
    y = y[y.index.isin(X.index)]
    if(model in reg_models):
        X_train, X_test, y_train, y_test = split_scale(X, y, transf, shuffle)
    else:
        X_train, X_test, y_train, y_test = bins_scale(X, y, transf, shuffle)
    res = run_sklearn_model(model, (X_train, y_train), (X_test, y_test), features, target)
    return(res)

def do_index(cur, model, transf = None, shuffle=False):
    index_cols = [x for x in index.columns if x[1] == cur[0] and x[2] == cur[1]]
    X = index[[col for col in index_cols if col[0] in features + ['Time features']]][:-1]
    y = index[[col for col in index_cols if col[0] in target]].shift(-1)[:-1]
    X = X.dropna(how='any')
    y = y[y.index.isin(X.index)]
    if(model in reg_models):
        X_train, X_test, y_train, y_test = split_scale(X, y, transf, shuffle)
    else:
        X_train, X_test, y_train, y_test = bins_scale(X, y, transf, shuffle)
    res = run_sklearn_model(model, (X_train, y_train), (X_test, y_test), features, target)
    return(res)

def do_forex_index(cur, model, transf = None, shuffle=False):
    index_cols = [x for x in index.columns if x[1] == cur]
    forex_cols = [x for x in forex.columns if x[1] == cur]
    index_X = index[[col for col in index_cols if col[0] in features + ['Time features']]][:-1]
    index_y = index[[col for col in index_cols if col[0] in target]].shift(-1)[:-1]
    forex_X = forex[[col for col in forex_cols if col[0] in features + ['Time features']]][:-1]
    forex_y = forex[[col for col in forex_cols if col[0] in target]].shift(-1)[:-1]
    index_X.columns = index_X.columns.values
    forex_X.columns = forex_X.columns.values
    X = pd.concat([index_X, forex_X], axis=1)
    y = pd.concat([index_y, forex_y], axis=1)
    y_cols = np.concatenate((index_y.columns.values, forex_y.columns.values))
    X = X.dropna(how='any')
    y = y[y.index.isin(X.index)]
    for y_col in (y_cols):
        print(y_col)
        y_temp = y[y_col]
        if(model in reg_models):
            X_train, X_test, y_train, y_test = split_scale(X, y_temp, transf, shuffle)
        else:
            X_train, X_test, y_train, y_test = bins_scale(X, y_temp, transf, shuffle)
        res = run_sklearn_model(model, (X_train, y_train), (X_test, y_test), features, target)
        print(res)


def get_cors_forex(cur, model, transf = None, shuffle=False):
    forex_cols = [x for x in forex.columns if x[1] == cur]
    X = forex[[col for col in forex_cols if col[0] in features + ['Time features']]][:-1]
    y = forex[[col for col in forex_cols if col[0] in target]].shift(-1)[:-1]
    X = X.dropna(how='any')
    y = y[y.index.isin(X.index)]
    if(model in reg_models):
        X_train, X_test, y_train, y_test = split_scale(X, y, transf, shuffle)
    else:
        X_train, X_test, y_train, y_test = bins_scale(X, y, transf, shuffle)
    # print(X)
    # print()
    corarr = (pd.DataFrame(X_train, columns = X.columns).join(pd.DataFrame(y_train).rename((lambda x: 'Target'), axis='columns')))
    print(corarr.corr()['Target'])

def get_cors_index(cur, model, transf = None, shuffle=False):
    index_cols = [x for x in index.columns if x[1] == cur[0] and x[2] == cur[1]]
    X = index[[col for col in index_cols if col[0] in features + ['Time features']]][:-1]
    y = index[[col for col in index_cols if col[0] in target]].shift(-1)[:-1]
    X = X.dropna(how='any')
    y = y[y.index.isin(X.index)]
    if(model in reg_models):
        X_train, X_test, y_train, y_test = split_scale(X, y, transf, shuffle)
    else:
        X_train, X_test, y_train, y_test = bins_scale(X, y, transf, shuffle)
    # print(X)
    # print()
    corarr = (pd.DataFrame(X_test, columns = X.columns).join(pd.DataFrame(y_test).rename((lambda x: 'Target'), axis='columns')))
    # corarr.to_csv("Lol_checking.csv")
    print(corarr.corr()['Target'])

# get_cors_index(index_pairs[0], LinearRegression, StandardScaler)
# print(forex_pairs)
# do_forex_index(forex_pairs[0], LinearRegression, None)
do_forex_index('BDT', LinearRegression)
# print(forex.columns[:18])
