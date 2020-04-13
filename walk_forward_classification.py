import pandas as pd
import numpy as np
from sklearn.linear_model import *
from sklearn.metrics import *
from sklearn.preprocessing import *
from sklearn.svm import *
import warnings
from sklearn.exceptions import DataConversionWarning

warnings.filterwarnings(action='ignore', category=DataConversionWarning)

forex = pd.read_csv('prep_forex.csv', header=[0, 1], index_col=0)
index = pd.read_csv('prep_index.csv', header=[0, 1, 2], index_col=0)

forex_pairs = list(set([x[1] for x in forex.columns if x[0] == 'Close']))
index_pairs = list(set([(x[1], x[2]) for x in index.columns if x[0] == 'Close']))

cls_models = [RidgeClassifier]
reg_models = [SVR, LinearSVR, LinearRegression]

forex_feature = "VND"
index_feature = ("MNT", "MNE Top 20")


metric = 'Close'
metrics = ['Open', 'Close', 'Low', 'High']
target = [metric + '_Ret']

forex_features = ['Intraday_HL', 'Intraday_OC', 'Prev_close_open'] + [y + x for x in
                                                                      ['_Ret', '_Ret_MA_3', '_Ret_MA_15', '_Ret_MA_45',
                                                                       '_MTD', '_YTD'] for y in metrics]
index_features = ['Intraday_HL', 'Intraday_OC', 'Prev_close_open'] + [y + x for x in
                                                                      ['_Ret', '_Ret_MA_3', '_Ret_MA_15', '_Ret_MA_45',
                                                                       '_MTD', '_YTD'] for y in (metrics + ['Volume'])]

scalers = [MinMaxScaler, StandardScaler, FunctionTransformer]

def run_sklearn_model(model, train, test, features, target):
    X_train, y_train = train
    X_test, y_test = test
    X_train = pd.DataFrame(X_train)
    X_test = pd.DataFrame(X_test)
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
    prediction = pd.DataFrame(prediction)
    print(model.__name__ + " " + str(period) + " accuracy = ", accuracy_score(y_true, prediction))
    print(confusion_matrix(y_true, prediction))
    x_true = pd.concat([X_train, X_test], axis=0)
    prediction.index = x_true.index
    return prediction


def split_scale(X, y, scaler, train_index, test_index, shuffle=False, poly=False):
    X_train, X_test = X.iloc[:train_index], X.iloc[train_index:]
    y_train, y_test = y.iloc[:train_index], y.iloc[train_index:]

    if scaler:
        scaler_X = scaler()
        if(scaler == FunctionTransformer):
            print("hi")
            scaler_X = scaler(np.log1p)
        scaler_X = scaler_X.fit(X_train)
        X_train = scaler_X.transform(X_train)
        X_test = scaler_X.transform(X_test)

    scaler_y = Binarizer()
    y_train = np.array(y_train).reshape(1, -1)
    y_test = np.array(y_test).reshape(1, -1)
    y_train = scaler_y.transform(y_train)
    y_test = scaler_y.transform(y_test)
    y_train[y_train == 0] = -1
    y_test[y_test == 0] = -1
    y_test = np.array(y_test).reshape(-1, 1)
    y_train = np.array(y_train).reshape(-1, 1)
    print(pd.DataFrame(X_test).shape)
    return (X_train, X_test, y_train, y_test)


def add_cross_domain_features(f_feat = forex_feature, i_feat = index_feature):
    index_cols = index_cols = [x for x in index.columns if x[1] == i_feat[0] and x[2] == i_feat[1]]
    X_i = index[[col for col in index_cols if col[0] in index_features + ['Time features']]][:-1]
    forex_cols = [x for x in forex.columns if x[1] == f_feat]
    X_f = forex[[col for col in forex_cols if col[0] in forex_features + ['Time features']]][:-1]
    cd_feats = pd.concat([X_i, X_f], axis=1)
    return(cd_feats)

def do_forex(cur, model, train_index, test_index, transf=None, shuffle=False, poly=False, transf_features_also=False):
    forex_cols = [x for x in forex.columns if x[1] == cur]
    X = forex[[col for col in forex_cols if col[0] in forex_features + ['Time features']]][:-1]
    X = X.join(add_cross_domain_features())
    y = forex[[col for col in forex_cols if col[0] in target]].shift(-1)[:-1]
    X = X.dropna(how='any')
    y = y[y.index.isin(X.index)]
    X_train, X_test, y_train, y_test = split_scale(X, y, transf, train_index, test_index, shuffle, poly)
    res = run_sklearn_model(model, (X_train, y_train), (X_test, y_test), forex_features, target)
    return (res)


def do_index(cur, model, train_index, test_index, transf=None, shuffle=False, poly=False, transf_features_also=False):
    index_cols = [x for x in index.columns if x[1] == cur[0] and x[2] == cur[1]]
    X = index[[col for col in index_cols if col[0] in index_features + ['Time features']]][:-1]
    X = X.join(add_cross_domain_features())
    y = index[[col for col in index_cols if col[0] in target]].shift(-1)[:-1]
    X = X.dropna(how='any')
    y = y[y.index.isin(X.index)]
    X_train, X_test, y_train, y_test = split_scale(X, y, transf, train_index, test_index, shuffle, poly)
    res = run_sklearn_model(model, (X_train, y_train), (X_test, y_test), index_features, target)
    return (res)


def iterate_markets():
    shuffle = False
    poly = False
    reg_res = pd.DataFrame()
    for f_m in ["BDT"]:
        print(f_m)
        for model in cls_models:
            for scaler in scalers:
                for i in range(30, 35, 5):
                    if (f_m in forex_pairs):
                        train_index = i
                        test_index = 0
                        res = do_forex(f_m, model, train_index, test_index, scaler, shuffle, poly)
                    else:
                        train_index = int(len(index) * .8)
                        test_index = 0
                        res = do_index(f_m, model, train_index, test_index, scaler, shuffle, poly)
                    if scaler:
                        res.columns = [scaler.__name__ + "_pred_" + str(i)]
                    else:
                        res.columns = ["None_pred_" + str(i)]

                    print(reg_res.shape, res.shape)
                    # reg_res = reg_res.append(res)
                    reg_res = pd.concat([reg_res, res], axis=1)

                    # except Exception as e:
                    #     print(e)
            reg_res.to_csv(str(f_m) + "_" + model.__name__ + "_sliding_sclaed_results.csv")


iterate_markets()
