import pandas as pd
import numpy as np
from sklearn.linear_model import *
from sklearn.metrics import *
from sklearn.preprocessing import *
from sklearn.svm import *
import warnings
from sklearn.exceptions import DataConversionWarning
from statsmodels.tsa.holtwinters import ExponentialSmoothing

warnings.filterwarnings(action='ignore', category=DataConversionWarning)

forex = pd.read_csv('prep_forex.csv', header=[0, 1], index_col=0)
index = pd.read_csv('prep_index.csv', header=[0, 1, 2], index_col=0)

forex_pairs = list(set([x[1] for x in forex.columns if x[0] == 'Close']))
index_pairs = list(set([(x[1], x[2]) for x in index.columns if x[0] == 'Close']))

cls_models = [SVC]
reg_models = [SVR, LinearSVR, LinearRegression]

metric = 'Close'
metrics = ['Open', 'Close', 'Low', 'High']
target = [metric + '_Ret']

forex_features = ['Intraday_HL', 'Intraday_OC', 'Prev_close_open'] + [y + x for x in
                                                                      ['_Ret', '_Ret_MA_3', '_Ret_MA_15', '_Ret_MA_45',
                                                                       '_MTD', '_YTD'] for y in metrics]
index_features = ['Intraday_HL', 'Intraday_OC', 'Prev_close_open'] + [y + x for x in
                                                                      ['_Ret', '_Ret_MA_3', '_Ret_MA_15', '_Ret_MA_45',
                                                                       '_MTD', '_YTD'] for y in (metrics + ['Volume'])]

scalers = [Binarizer]


def run_sklearn_model(model, train, test, features, target):
    X_train, y_train = train
    X_test, y_test = test

    period = len(X_train)
    y_train, y_test = y_train.astype(int), y_test.astype(int)
    prediction = []
    data = X_train.values
    data_y = y_train
    for i, t in enumerate(X_test.values):
        reg = model()
        reg.fit(data, data_y)
        if i == 0:
            y = reg.predict(X_train)
            for elem in y:
                prediction.append(elem)
        y = reg.predict([t])
        prediction.append(y[0])
        data = np.vstack((data, t))
        data = np.delete(data, 0, 0)
        data_y = np.append(data_y, y_test[i])
        data_y = np.delete(data_y, 0, 0)
    y_true = np.vstack((y_train, y_test))
    print(model.__name__ + " " + str(period) + " accuracy = ", accuracy_score(y_true, prediction))
    print(confusion_matrix(y_true, prediction))
    pred = pd.DataFrame(prediction)
    x_true = pd.concat([X_train, X_test], axis=0)
    pred.index = x_true.index
    return pred


def split_scale(X, y, scaler, train_index, test_index, shuffle=False, poly=False, transf_features_also=False):
    X_train, X_test = X.iloc[:train_index], X.iloc[train_index:]
    y_train, y_test = y.iloc[:train_index], y.iloc[train_index:]
    if scaler:

        scaler_X = scaler()
        if (transf_features_also):
            X_train = scaler_X.transform(X_train)
            X_test = scaler_X.transform(X_test)
        scaler_y = scaler()

        y_train = np.array(y_train).reshape(1, -1)
        y_test = np.array(y_test).reshape(1, -1)
        y_train = scaler_y.transform(y_train)
        y_test = scaler_y.transform(y_test)
        y_train[y_train == 0] = -1
        y_test[y_test == 0] = -1
        y_test = np.array(y_test).reshape(-1, 1)
        y_train = np.array(y_train).reshape(-1, 1)

    return (X_train, X_test, y_train, y_test)


def do_forex(cur, model, train_index, test_index, transf=None, shuffle=False, poly=False, transf_features_also=False):
    forex_cols = [x for x in forex.columns if x[1] == cur]
    X = forex[[col for col in forex_cols if col[0] in forex_features + ['Time features']]][:-1]
    y = forex[[col for col in forex_cols if col[0] in target]].shift(-1)[:-1]
    X = X.dropna(how='any')
    y = y[y.index.isin(X.index)]
    X_train, X_test, y_train, y_test = split_scale(X, y, transf, train_index, test_index, shuffle, poly,
                                                   transf_features_also)
    res = run_sklearn_model(model, (X_train, y_train), (X_test, y_test), forex_features, target)
    return (res)


def do_index(cur, model, train_index, test_index, transf=None, shuffle=False, poly=False, transf_features_also=False):
    index_cols = [x for x in index.columns if x[1] == cur[0] and x[2] == cur[1]]
    X = index[[col for col in index_cols if col[0] in index_features + ['Time features']]][:-1]
    y = index[[col for col in index_cols if col[0] in target]].shift(-1)[:-1]
    X = X.dropna(how='any')
    y = y[y.index.isin(X.index)]
    X_train, X_test, y_train, y_test = split_scale(X, y, transf, train_index, test_index, shuffle, poly,
                                                   transf_features_also)
    res = run_sklearn_model(model, (X_train, y_train), (X_test, y_test), index_features, target)
    return (res)


# def walk_forward(train, test):
#     try:
#         X_train, y_train = train
#         X_test, y_test = test
#     except:
#         pass
#     prediction = []
#     data = X_train.values
#
#     for i, t in enumerate(X_test.values):
#         model = (ExponentialSmoothing(data).fit())
#         y = model.predict()
#         prediction.append(1 if y[0] > 0.001 else 0)
#         data = np.append(data, t)
#
#     pred = pd.DataFrame(prediction)
#     print(confusion_matrix(y_test, pred))
#     y_test = pd.DataFrame(y_test)
#     pred = pd.concat([pred, y_test], axis=1)
#     pred.index = X_test.index
#     return pred


def iterate_markets():
    shuffle = False
    poly = False
    transf_features_also = False
    for f_m in forex_pairs:
        print(f_m)
        for model in cls_models:
            for scaler in scalers:
               for i in range(30, 55, 5):
                    try:
                        if (f_m in forex_pairs):
                            train_index = i
                            test_index = 0
                            res = do_forex(f_m, model, train_index, test_index, scaler, shuffle, poly,
                                           transf_features_also)
                            res.columns = ["pred"]
                        else:
                            train_index = int(len(index) * .8)
                            test_index = 0
                            res = do_index(f_m, model, train_index, test_index, scaler, shuffle, poly,
                                           transf_features_also)
                            reg_res = res
                            reg_res.columns = ["pred"]
                        res.to_csv(str(f_m) + "_" + model.__name__ + "_sliding_" + str(i) + "_results.csv")
                    except Exception as e:
                        print(e)



iterate_markets()
