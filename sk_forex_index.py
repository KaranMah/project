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
from sklearn.naive_bayes import *
from sklearn.cross_decomposition import PLSRegression
from sklearn.tree import *
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import *
from sklearn.metrics import *

import warnings
warnings.filterwarnings("ignore")

forex = pd.read_csv('prep_forex.csv', header=[0,1], index_col=0)
index = pd.read_csv('prep_index.csv', header=[0,1,2], index_col=0)

forex_pairs = list(set([x[1] for x in forex.columns if x[0] == 'Close']))
index_pairs = list(set([(x[1], x[2]) for x in index.columns if x[0] == 'Close']))

reg_models = [LinearRegression, Ridge, RidgeCV, Lasso, LassoCV, LassoLarsCV, LassoLarsIC, MultiTaskLasso,
              ElasticNet, ElasticNetCV, MultiTaskElasticNet, MultiTaskElasticNetCV, LassoLars,
              OrthogonalMatchingPursuit, BayesianRidge, ARDRegression,  LogisticRegression, LogisticRegressionCV,
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

reg_scalers = [None, MinMaxScaler, MaxAbsScaler, StandardScaler, RobustScaler, Normalizer,
            QuantileTransformer, PowerTransformer, FunctionTransformer]

cls_scalers = [Binarizer, KBinsDiscretizer]

metric = 'Close'
metrics = ['Open', 'Close', 'Low', 'High', 'Volume']
target = [metric+'_Ret']
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
    deets = {}
    if(model in cls_models):
        deets['F1'] = f1_score(y_test, y_pred)
        deets['Precision'] = precision_score(y_test, y_pred)
        deets['Recall'] = recall_score(y_test, y_pred)
        deets['AUC'] = roc_auc_score(y_test, reg.fit(X_train, y_train).decision_function(X_test))
    try:
        deets['MSE'] = mean_squared_error(y_test, y_pred)
        deets['R2'] = r2_score(y_test, y_pred)
        return(deets)
    except:
        y_pred = ([x[0] for x in y_pred])
        for i in range(len(y_pred)):
            if(np.isnan(y_pred[i])):
                y_pred[i] = 0
        deets['MSE'] = mean_squared_error(y_test, y_pred)
        deets['R2'] = r2_score(y_test, y_pred)
        return(deets)
    # print("MSE: ", mean_squared_error(y_test, y_pred))
    # print("R2: ", r2_score(y_test, y_pred))
    # plot_results(y_test, y_pred, model)


def split_scale(X, y, scaler, shuffle=False, poly=False):
    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=shuffle, test_size=0.2)
    if(poly):
        X_train = PolynomialFeatures(2).fit_transform(X_train)
        X_test = PolynomialFeatures(2).fit_transform(X_test)
    if(scaler):
        scaler_X = scaler()
        if(scaler == reg_scalers[-1]):
            scaler_X = scaler(np.log1p)
        scaler_X = scaler_X.fit(X_train)
        X_train = scaler_X.transform(X_train)
        X_test = scaler_X.transform(X_test)
        scaler_y = scaler()
        if(scaler == reg_scalers[-1]):
            scaler_y = scaler(np.log1p)
        try:
            scaler_y = scaler_y.fit(y_train.values.reshape(-1,1))
            y_train = scaler_y.transform(y_train.values.reshape(-1, 1))
        except:
            y_train = np.nan_to_num(y_train, posinf=1, neginf=-1)
            scaler_y = scaler_y.fit(y_train.reshape(-1,1))
            y_train = scaler_y.transform(y_train.reshape(-1, 1))
        try:
            y_test = scaler_y.transform(y_test.values.reshape(-1, 1))
        except:
            y_test = np.nan_to_num(y_test, posinf=1, neginf=-1)
            y_test = scaler_y.transform(y_test.reshape(-1, 1))
    return(X_train, X_test, y_train, y_test)


def do_forex_index(cur, model, transf = None, shuffle=False, poly=False):
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
    X = X.dropna(how='all', axis=1)
    X = X.dropna(how='any')
    y = y[y.index.isin(X.index)]
    tot_res = []
    for y_col in (y_cols):
        y_temp = y[y_col]
        res = []
        if((model in reg_models and transf in reg_scalers) or (model in cls_models and transf in cls_scalers)):
            X_train, X_test, y_train, y_test = split_scale(X, y_temp, transf, shuffle, poly)
            try:
                res = run_sklearn_model(model, (X_train, y_train), (X_test, y_test), features, target)
                res['Pair'] = y_col
                res['Transformation'] = transf().__class__.__name__ if transf is not None else None
                res['Shuffle'] = shuffle
                res['Model'] = model().__class__.__name__
                res['Poly'] = poly
                tot_res.append(res)
            except:
                continue
                # res = run_sklearn_model(model, (X_train, y_train), (X_test, y_test), features, target)
            # print(res)
    return(tot_res)

def iterate_markets():
    reg_res = []
    for f_m in forex_pairs:
        print(f_m)
        for model in (reg_models + cls_models):
            for scaler in (reg_scalers + cls_scalers):
                for shuffle in [True, False]:
                    for poly in [True, False]:
                        res = do_forex_index(f_m, model, scaler, shuffle, poly)
                        print(res)
                        reg_res.extend(res)
    return(reg_res)

res = iterate_markets()
res_df = pd.DataFrame(res, columns= ['Pair', 'Model', 'Transformation', 'Shuffle', 'Poly', 'MSE', 'R2', 'F1', 'Precision', 'Recall', 'AUC'])
# print(res_df)
res_df.to_csv("sk_forex_index_regression.csv")
# do_stuff(["HKD", "Hang Seng"], LinearRegression)
