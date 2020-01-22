

###############################################################################
###############################################################################

ax = sns.heatmap(
    open_forex.corr(),
    vmin=-1, vmax=1, center=0,
    cmap=sns.diverging_palette(20, 220, n=200),
    square=True)


###############################################################################
###############################################################################

def plot_results(y_true, y_pred, model):
    plot_df = pd.concat([y_true.reset_index(drop=True), pd.Series(y_pred)], axis=1, ignore_index=True)
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
    print("MSE: ", mean_squared_error(y_test, y_pred))
    print("R2: ", r2_score(y_test, y_pred))
    plot_results(y_test, y_pred, model)


def do_stuff_forex(raw_data, cur, periods, model):
    data = raw_data[cur]
    data = data.reset_index()
    data['ret'] = (data[[cur]] - data[[cur]].shift(1))/(data[[cur]].shift(1))
    for i in periods:
            data['ret_'+str(i)] = data['ret'].rolling(i).mean().shift()
    data = data.iloc[max(periods)+1:,:]
    plt.plot(data['ret'])
    plot_acf(data['ret'])
    plot_pacf(data['ret'])
    adf_res = adfuller(data['ret'])
    print("ADF statistic: ", adf_res[0])
    print("P - Value: ", adf_res[1])
    print("Lag: ", adf_res[2])
    features = list(map(lambda x: "ret_"+str(x), periods))
    X_train, X_test, y_train, y_test = train_test_split(data[features], data['ret'], test_size=0.33)
    run_sklearn_model(model, (X_train, y_train), (X_test, y_test), features, 'ret')

do_stuff_forex(open_forex, "INR", [1, 3, 30], RandomForestRegressor)
