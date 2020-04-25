from os import listdir
from os.path import join, isfile

import pandas as pd

def get_daily_values(data, daily_scores=pd.DataFrame()):
    r = data.resample('D')
    daily_scores['Average'] = r.mean()['test_score']
    daily_scores['Max'] = r.max()['test_score']
    daily_scores['Min'] = r.min()['test_score']
    daily_scores['Std'] = r.std()['test_score']
    daily_scores['Variance'] = r.var()['test_score']
    daily_scores['Count'] = r.count()['test_score']
    return daily_scores


def aggregate_weekends(daily_scores):
    final_values = pd.DataFrame()
    df = pd.DataFrame()
    df['Date'] = daily_scores.index
    for i in range(len(daily_scores)):
        day = df['Date'].iloc[i].weekday()
        if day == 4:
            weekend_values = daily_scores.iloc[i:i+3]
            weekend_values = weekend_values.dropna()
            temp = {}
            temp['Average'] = weekend_values['Average'].mean()
            temp['Max'] = weekend_values['Max'].max()
            temp['Min'] = weekend_values['Min'].min()
            temp['Std'] = weekend_values['Std'].std()
            temp['Variance'] = weekend_values['Variance'].min()
            temp['Count'] = weekend_values['Count'].sum()
            temp = pd.DataFrame(temp, index=[df.iloc[i].get('Date')])
            final_values = final_values.append(temp)
        elif day == 5 or day == 6:
            continue
        else:
            final_values = final_values.append(daily_scores.iloc[i])
    return final_values


def get_results(tag):
    file_list_with_same_tag = [f for f in allFiles if tag in f]
    daily_scores = pd.DataFrame()
    df = pd.DataFrame(columns=['test_score'])
    for file in file_list_with_same_tag:
        res = pd.read_json(RESULTS_PATH + file, lines=True)
        temp = pd.DataFrame()
        res.sort_index()
        temp['test_score'] = pd.to_numeric(res['test_score'].str.get(0))
        temp.index = res[timestamp]
        df = pd.concat([df, temp])
    get_daily_values(df, daily_scores)
    final_values = aggregate_weekends(daily_scores)
    return final_values


def save_results(tag, score):
    csv_name = AGGREGATE_PATH + tag + "_aggregated_data.csv"
    score.to_csv(csv_name)


RESULTS_PATH = "/data/results/"
AGGREGATE_PATH = "/data/aggregate/"
hashtags = []

#countryList = ['Pakistan', 'Mongolia', 'Bangladesh', 'SriLanka', 'Karachi', 'Dhaka', 'Ulaanbaatar', 'Colombo']
countryList = ["#Ulaan"]
allFiles = [f for f in listdir(RESULTS_PATH) if (isfile(join(RESULTS_PATH, f)) and any(j in f for j in countryList))]
allFiles = ["#Ulaanbaatar_since_2010-01-01_until_2019-12-31.json"]
allFiles.sort()

for file_name in allFiles:
    print(file_name)
    name = file_name.split('_')
    if name[0] == "r":
        newTag = name[1]
        timestamp = 'timestamp'
    else:
        newTag = name[0]
        timestamp = 'datetime'
    scores = get_results(newTag)

    save_results(newTag, scores)

        














