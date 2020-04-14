import pandas as pd
import numpy as np
from sklearn.linear_model import *
from sklearn.metrics import *
from sklearn.preprocessing import *
from sklearn.svm import *
import warnings
from sklearn.exceptions import DataConversionWarning

warnings.filterwarnings(action='ignore', category=DataConversionWarning)
warnings.filterwarnings(action='ignore', category=UserWarning)

forex = pd.read_csv('prep_forex.csv', header=[0, 1], index_col=0)
index = pd.read_csv('prep_index.csv', header=[0, 1, 2], index_col=0)

forex_pairs = list(set([x[1] for x in forex.columns if x[0] == 'Close']))
index_pairs = list(set([(x[1], x[2]) for x in index.columns if x[0] == 'Close']))

cls_models = [RidgeClassifierCV, RidgeClassifier]
reg_models = [SVR, LinearSVR, LinearRegression]

target_markets = [('PKR', 'Karachi 100'), ('LKR', 'CSE All-Share')]
features = {"MNT": [None, "LKR", ("NZD", "NZX MidCap")],
            ('PKR', 'Karachi 100'): [None, "INR", ('JPY', 'NIkkei 225')],
            ('LKR', 'CSE All-Share'): [None, "IDR", ('MNT', 'MNE Top 20')],
            "BDT": [("IDR", "IDX Composite"), None, "VND"]}

# forex_feature = "VND"
# index_feature = ("BDT", "DSE 30")

accuracy_lst = pd.DataFrame()
csv_name = ""
csv_dir = "./walk_forward_sliding_cross_results/"

metric = 'Close'
metrics = ['Open', 'Close', 'Low', 'High']
target = [metric + '_Ret']

forex_features = ['Intraday_HL', 'Intraday_OC', 'Prev_close_open'] + [y + x for x in
                                                                      ['_Ret', '_Ret_MA_3', '_Ret_MA_15', '_Ret_MA_45',
                                                                       '_MTD', '_YTD'] for y in metrics]
index_features = ['Intraday_HL', 'Intraday_OC', 'Prev_close_open'] + [y + x for x in
                                                                      ['_Ret', '_Ret_MA_3', '_Ret_MA_15', '_Ret_MA_45',
                                                                       '_MTD', '_YTD'] for y in (metrics + ['Volume'])]

# scalers = [None, MinMaxScaler, StandardScaler]
xstr = lambda s: '' if s is None else str(s)

def run_sklearn_model(model, train, test, feat, target):
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
        global accuracy_lst
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

    acc = accuracy_score(y_true, prediction)
    print(model.__name__ + " accurcy for "+ xstr(feat) + " with period "+ str(period)+"="+str(acc))
    class_report = classification_report(y_true, prediction, digits=2)
    data = np.array([class_report])
    acc_df = pd.DataFrame(data=data)
    accuracy_lst = pd.concat([accuracy_lst, acc_df])

    x_true = pd.concat([X_train, X_test], axis=0)
    prediction.index = x_true.index
    return prediction

def split_scale(X, y, scaler, train_index, test_index, shuffle=False, poly=False):
    X_train, X_test = X.iloc[:train_index], X.iloc[train_index:]
    y_train, y_test = y.iloc[:train_index], y.iloc[train_index:]
    if scaler is not None:
        scaler_X = scaler()
        if(scaler == FunctionTransformer):
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
    return (X_train, X_test, y_train, y_test)


def add_cross_domain_features(feat):
    if(isinstance(feat, tuple)):
        index_cols = index_cols = [x for x in index.columns if x[1] == feat[0] and x[2] == feat[1]]
        X = index[[col for col in index_cols if col[0] in index_features + ['Time features']]][:-1]
    else:
        forex_cols = [x for x in forex.columns if x[1] == feat]
        X = forex[[col for col in forex_cols if col[0] in forex_features + ['Time features']]][:-1]
    return(X)


def do_forex(cur, model, train_index, test_index, feat, transf=None, shuffle=False, poly=False):
    forex_cols = [x for x in forex.columns if x[1] == cur]
    X = forex[[col for col in forex_cols if col[0] in forex_features + ['Time features']]][:-1]
    if feat:
        X = X.join(add_cross_domain_features(feat))

    y = forex[[col for col in forex_cols if col[0] in target]].shift(-1)[:-1]
    X = X.dropna(how='all', axis=1)
    X = X.dropna(how='any')
    y = y[y.index.isin(X.index)]

    X_train, X_test, y_train, y_test = split_scale(X, y, transf, train_index, test_index, shuffle, poly)
    res = run_sklearn_model(model, (X_train, y_train), (X_test, y_test), feat, target)
    return res


def do_index(cur, model, train_index, test_index, feat, transf=None, shuffle=False, poly=False):
    index_cols = [x for x in index.columns if x[1] == cur[0] and x[2] == cur[1]]
    X = index[[col for col in index_cols if col[0] in index_features + ['Time features']]][:-1]
    if feat:
        X = X.join(add_cross_domain_features(feat))
    y = index[[col for col in index_cols if col[0] in target]].shift(-1)[:-1]
    X = X.dropna(how='all', axis=1)
    X = X.dropna(how='any')
    y = y[y.index.isin(X.index)]
    X_train, X_test, y_train, y_test = split_scale(X, y, transf, train_index, test_index, shuffle, poly)
    res = run_sklearn_model(model, (X_train, y_train), (X_test, y_test), feat, target)
    return (res)


def iterate_markets():
    shuffle = False
    poly = False
    global accuracy_lst
    for f_m in target_markets:
        print(f_m)
        for model in cls_models:
            print(model.__name__)
            for scaler in [None]:
                for feat in features[f_m]:
                    print(feat)
                    reg_res = pd.DataFrame()
                    rows = []
                    for i in range(5, 55, 5):
                        try:
                            if (f_m in forex_pairs):
                                train_index = i
                                test_index = 0
                                res = do_forex(f_m, model, train_index, test_index, feat, scaler, shuffle, poly)
                            else:
                                train_index = i
                                test_index = 0
                                res = do_index(f_m, model, train_index, test_index, feat, scaler, shuffle, poly)
                            if isinstance(feat, tuple):
                                res.columns = [feat[0] + "_pred_" + str(i)]
                            else:
                                res.columns = [xstr(feat) + "_pred_" + str(i)]

                            reg_res = pd.concat([reg_res, res], axis=1)
                            rows.append(i)
                        except Exception as e:
                            print(e)
                            pass

                    csv_name = csv_dir + str(f_m) + "_" + model.__name__ + "_" + xstr(feat)
                    reg_res.to_csv(csv_name+".csv")
                    accuracy_lst.index = rows
                    accuracy_lst.columns = ["report"]
                    accuracy_lst.to_csv(csv_name + "_metrics.csv")
                    accuracy_lst = pd.DataFrame()


iterate_markets()

