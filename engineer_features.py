import json
from os import listdir, makedirs
from os.path import isfile, join, exists
import numpy as np
import csv
from datetime import datetime as dt


def get_daily_values(data, daily_scores):
    counter = 0
    for key in data:
        daily_scores[counter].update({'Average': np.average(data[key])})
        daily_scores[counter].update({'Max': max(data[key])})
        daily_scores[counter].update({'Min': min(data[key])})
        daily_scores[counter].update({'Standard Deviation': np.std(data[key])})
        daily_scores[counter].update({'Variance': np.var(data[key])})
        daily_scores[counter].update({'Positive Multiplied': np.product(data[key])})
        daily_scores[counter].update(({'Count': len(data[key])}))
        counter += 1


def aggregate_weekends(daily_scores):
    i = 0
    j = 0
    final_values = []
    while i < len(daily_scores):
        day = dt.strptime(daily_scores[i]["Date"], '%d-%m-%Y').weekday()
        if day == 4:
            weekend_values = daily_scores[i:i+3]
            final_values.append({'Date': daily_scores[i]["Date"]})
            final_values[j].update({'Average': np.average([d['Average'] for d in weekend_values])})
            final_values[j].update({'Max': max([d['Max'] for d in weekend_values])})
            final_values[j].update({'Min': min([d['Min'] for d in weekend_values])})
            final_values[j].update({'Standard Deviation': np.std([d['Standard Deviation'] for d in weekend_values])})
            final_values[j].update({'Variance': np.var([d['Variance'] for d in weekend_values])})
            final_values[j].update({'Count': sum([d['Count'] for d in weekend_values])})
        elif day == 5 or day == 6:
            i += 1
            continue
        else:
            final_values.append(daily_scores[i])
        i += 1
        j += 1
    return final_values


def get_results(tag):
    file_list_with_same_tag = [f for f in allFiles if tag in f]
    daily_scores = []
    test_scores = {}
    for file in file_list_with_same_tag:
        with open(RESULTS_PATH + "/" + file, 'r') as f:
            for line in f:
                obj = json.loads(line)
                if obj[timestamp] in test_scores:
                    test_scores[f'{obj[timestamp]}'].append(float(obj['test_score'][0]))
                else:
                    test_scores[f'{obj[timestamp]}'] = [float(obj['test_score'][0])]
                    if timestamp == "timestamp":
                        daily_scores.append({'Date': dt.strptime(obj[timestamp], '%d-%m-%Y')})
                    else:
                        date = obj[timestamp].split(' ')[0]
                        daily_scores.append({'Date': dt.strptime(date, '%Y-%m-%d')})

    get_daily_values(test_scores, daily_scores)
    final_values = aggregate_weekends(daily_scores)
    return final_values


def save_results(tag):
    csv_columns = ['Date', 'Average', 'Max', 'Min', 'Standard Deviation', 'Variance', 'Positive Multiplied', 'Count']
    csv_name = "/data/aggregate/" + tag + "_aggregated_data.csv"
    try:
        with open(csv_name, 'w') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=csv_columns)
            writer.writeheader()
            for data in scores:
                writer.writerow(data)
    except IOError:
        print("couldn't write into  file")


RESULTS_PATH = "/data/results"
hashtags = []


allFiles = [f for f in listdir(RESULTS_PATH) if isfile(join(RESULTS_PATH, f))]

for file_name in allFiles:
    name = file_name.split('_')
    if name[0] == "r":
        newTag = name[1]
        timestamp = 'timestamp'
    else:
        newTag = name[0]
        timestamp = 'datetime'

    if newTag not in hashtags:
        hashtags.append(newTag)
        scores = get_results(newTag)
        save_results(newTag)














