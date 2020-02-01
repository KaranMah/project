import json
import numpy as np
import pandas as pd


def interpolate_data(save=False):
    forex = pd.read_csv("historical_forex.csv", parse_dates=["Date"])
    pivoted_forex = pd.pivot_table(forex, index=["Date"], columns=["Currency"], values=["Open", "High", "Low", "Close"])
    first_date = min(pivoted_forex.index)
    last_date = max(pivoted_forex.index)
    temp_dates = list(filter(lambda x: x.weekday() not in [5,6], pd.date_range(first_date, last_date)))
    pivoted_forex = pivoted_forex.reindex(temp_dates)
    for i in range(len(pivoted_forex.columns)):
        pivoted_forex.iloc[:,i] = pivoted_forex.iloc[:,i].interpolate(method="time")
    if(save):
        pivoted_forex.to_csv('interpolated_historical_forex.csv')

    index = pd.read_csv("historical_index.csv", parse_dates=["Date"])
    pivoted_index = pd.pivot_table(index, index=["Date"], columns=["Currency", "Idx"], values=["Open", "High", "Low", "Close", "Volume"])
    first_date = min(pivoted_index.index)
    last_date = max(pivoted_index.index)
    temp_dates = list(filter(lambda x: x.weekday() not in [5,6], pd.date_range(first_date, last_date)))
    pivoted_index = pivoted_index.reindex(temp_dates)
    for i in range(len(pivoted_index.columns)):
        pivoted_index.iloc[:,i] = pivoted_index.iloc[:,i].interpolate(method="time")
    if(save):
        pivoted_index.to_csv('interpolated_historical_index.csv')

    return (pivoted_forex, pivoted_index)

def oc_return(data, index = True):
    intraday = ((data['Close']-data['Open'])/data['Open'])
    if(index):
        intraday.columns = pd.MultiIndex.from_tuples([('Intraday_OC',)+x for x in intraday.columns])
    else:
        intraday.columns = pd.MultiIndex.from_tuples([('Intraday_OC',x) for x in intraday.columns])
    concat_data = data.join(intraday)
    return(concat_data)

def hl_return(data, index = True):
    intraday = ((data['High']-data['Low'])/data['Low'])
    if(index):
        intraday.columns = pd.MultiIndex.from_tuples([('Intraday_HL',)+x for x in intraday.columns])
    else:
        intraday.columns = pd.MultiIndex.from_tuples([('Intraday_HL',x) for x in intraday.columns])
    concat_data = data.join(intraday)
    return(concat_data)

def prev_close_open_return(data, index = True):
    close_open = ((data['Open']-(data['Close'].shift(1)))/(data['Close'].shift(1)))
    if(index):
        close_open.columns = pd.MultiIndex.from_tuples([('Prev_close_open',)+x for x in close_open.columns])
    else:
        close_open.columns = pd.MultiIndex.from_tuples([('Prev_close_open',x) for x in close_open.columns])
    concat_data = data.join(close_open)
    return(concat_data)

def get_returns(data, index=True):
    concat_data = data
    raws = ['Open', 'High', 'Low', 'Close', 'Volume']
    for met in raws:
        try:
            temp = data[met].pct_change(1)
        except:
            continue
        if(index):
            temp.columns = pd.MultiIndex.from_tuples([(met+'_Ret',)+x for x in temp.columns])
        else:
            temp.columns = pd.MultiIndex.from_tuples([(met+'_Ret',x) for x in temp.columns])
        concat_data = concat_data.join(temp)
    return(concat_data)

def moving_average(data, index=True):
    concat_data = data
    raws = ['Open_Ret', 'High_Ret', 'Low_Ret', 'Close_Ret', 'Volume_Ret']
    for met in raws:
        for t in [3,15,45]:
            try:
                temp = data[met].rolling(t).mean()
            except:
                continue
            if(index):
                temp.columns = pd.MultiIndex.from_tuples([(met+'_MA_'+str(t),)+x for x in temp.columns])
            else:
                temp.columns = pd.MultiIndex.from_tuples([(met+'_MA_'+str(t),x) for x in temp.columns])
            concat_data = concat_data.join(temp)
    return(concat_data)

def get_period_to_date(data, index=True):
    if(index):
        concat_data = data
        concat_mtd = data.groupby([data.index.year, data.index.month]).transform('first')
        temp_mtd = data[['Open', 'High', 'Low', 'Close', 'Volume']].divide(concat_mtd[['Open', 'High', 'Low', 'Close', 'Volume']]) - 1
        col_names = (list(set([x + '_MTD' for x in temp_mtd.columns.get_level_values(0)])))
        temp_mtd.columns = pd.MultiIndex.from_tuples([(x[0] + '_MTD',)+x[1:] for x in temp_mtd.columns])
        concat_data = concat_data.join(temp_mtd)

        concat_ytd = data.groupby([data.index.year]).transform('first')
        temp_ytd = data[['Open', 'High', 'Low', 'Close', 'Volume']].divide(concat_ytd[['Open', 'High', 'Low', 'Close', 'Volume']]) - 1
        col_names = (list(set([x + '_MTD' for x in temp_ytd.columns.get_level_values(0)])))
        temp_ytd.columns = pd.MultiIndex.from_tuples([(x[0] + '_YTD',)+x[1:] for x in temp_ytd.columns])
        concat_data = concat_data.join(temp_ytd)
    else:
        concat_data = data
        concat_mtd = data.groupby([data.index.year, data.index.month]).transform('first')
        temp_mtd = data[['Open', 'High', 'Low', 'Close']].divide(concat_mtd[['Open', 'High', 'Low', 'Close']]) - 1
        col_names = (list(set([x + '_MTD' for x in temp_mtd.columns.get_level_values(0)])))
        temp_mtd.columns = pd.MultiIndex.from_tuples([(x[0] + '_MTD',)+x[1:] for x in temp_mtd.columns])
        concat_data = concat_data.join(temp_mtd)

        concat_ytd = data.groupby([data.index.year]).transform('first')
        temp_ytd = data[['Open', 'High', 'Low', 'Close']].divide(concat_ytd[['Open', 'High', 'Low', 'Close']]) - 1
        col_names = (list(set([x + '_MTD' for x in temp_ytd.columns.get_level_values(0)])))
        temp_ytd.columns = pd.MultiIndex.from_tuples([(x[0] + '_YTD',)+x[1:] for x in temp_ytd.columns])
        concat_data = concat_data.join(temp_ytd)
    return(concat_data)


def add_date_values(data, index=True):
    concat_data = data
    temp = {'Weekday': concat_data.index.weekday, 'Year': concat_data.index.year,
            'Month': concat_data.index.month, 'Day': concat_data.index.day}
    temp = pd.DataFrame(temp, index=concat_data.index)
    if(index):
        print(temp.columns)
        temp.columns = pd.MultiIndex.from_tuples([('Time features',x,x) for x in temp.columns])
    else:
        temp.columns = pd.MultiIndex.from_tuples([('Time features',x) for x in temp.columns])
    concat_data = concat_data.join(temp)
    return(concat_data)

def add_features(forex= None, index= None, save=False):
    forex = (pd.read_csv("interpolated_historical_forex.csv", header=[0,1], index_col=0)) if forex is None else forex
    transf_forex = oc_return(forex, False)
    transf_forex = hl_return(transf_forex, False)
    transf_forex = prev_close_open_return(transf_forex, False)
    transf_forex = get_returns(transf_forex, False)
    transf_forex = moving_average(transf_forex, False)
    transf_forex = add_date_values(transf_forex, False)
    transf_forex = get_period_to_date(transf_forex, False)
    if(save):
        transf_forex.to_csv('extra_features_forex.csv')

    index = pd.read_csv("interpolated_historical_index.csv", header=[0,1,2], index_col=0) if index is None else index
    transf_index = oc_return(index, True)
    transf_index = hl_return(transf_index, True)
    transf_index = prev_close_open_return(transf_index, True)
    transf_index = get_returns(transf_index, True)
    transf_index = moving_average(transf_index, True)
    transf_index = add_date_values(transf_index, True)
    transf_index = get_period_to_date(transf_index, False)
    if(save):
        transf_index.to_csv('extra_features_index.csv')

    return(transf_forex, transf_index)

def transform(forex= None, index= None, start_date = '01-01-2010', end_date = '12-31-2019', save=True):
    dates = list(filter(lambda x: x.weekday() not in [5,6], pd.date_range(start_date, end_date)))

    forex = (pd.read_csv("extra_features_forex.csv", header=[0,1], index_col=0)) if forex is None else forex
    ret_forex = forex[forex.index.isin(dates)]
    if(save):
        ret_forex.to_csv('prep_forex.csv')

    index = (pd.read_csv("extra_features_index.csv", header=[0,1], index_col=0)) if index is None else index
    ret_index = index[index.index.isin(dates)]
    if(save):
        ret_index.to_csv('prep_index.csv')
    return(ret_forex, ret_index)

def main():
    print("Interpolating forex and index rates...")
    try:
        interpolated_forex, interpolated_index = interpolate_data()
        print("Done interpolating...")
    except:
        print("Some error happened in interpolation...")
    print("Adding extra features...")
    try:
        updated_forex, updated_index = add_features(interpolated_forex, interpolated_index)
        print("Done updating features...")
    except:
        print("Some error in adding features...")
    print("Transforming dates...")
    try:
        prep_forex, prep_index = transform(updated_forex, updated_index)
        print("Done transforming for dates...")
    except:
        print("Some error in transforming dates...")

main()
