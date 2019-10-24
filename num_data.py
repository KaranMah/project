import json
import investpy
import numpy as np
import pandas as pd
import datetime as datetime

config = {
    "raw_data": "./currencies.txt",
    "cur_pairs": "./cur_pairs.txt",
    "earliest_date" : '01/01/2010',
    "latest_date": datetime.datetime.now().strftime("%d/%m/%Y")
}

def read_file(filename):
    dataset = pd.read_csv(filename, sep=';')
    dataset["Index"] = dataset["Index"].apply(lambda x: x.split(','))
    return dataset


def get_currency_pairs(currencies):
    lol = [(currencies[0] + "/" + x) for x in currencies[1:]]
    with open('cur_pairs.txt', 'w') as f:
        for item in lol:
            f.write("%s\n" % item)


def get_forex_rates(filename):

    pairs = list(pd.read_csv(filename, header=None)[0])
    final_df = pd.DataFrame()
    for pair in pairs:
        df = investpy.get_currency_cross_historical_data(currency_cross=pair,
                                                 from_date=config['earliest_date'],
                                                 to_date=config['latest_date'])
        final_df = final_df.append(df)
    final_df.to_csv("historical_forex.csv")

def get_index_rates(data):
    final_df = pd.DataFrame()
    for _, row in data.iterrows():
        for idx in row.Index:
            df = investpy.get_index_historical_data(index=idx,
                                        country=row.Country,
                                        from_date=config['earliest_date'],
                                        to_date=config['latest_date'])
            final_df = final_df.append(df)
    final_df.to_csv("historical_index.csv")

# if __name__ == '__main__':
data = read_file(config["raw_data"])
# get_currency_pairs(list(data["Currency"]))
# get_forex_rates(config["cur_pairs"])
get_index_rates(data[['Country', 'Index']])
