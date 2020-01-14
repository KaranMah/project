import json
from os import listdir
from os.path import isfile, join
import ast
import shutil

RESULTS_PATH = "/data/results"

allFiles = [f for f in listdir(RESULTS_PATH) if isfile(join(RESULTS_PATH, f))]

json_file_name = allFiles[0]

data = []
timestamp = 'timestamp'
dailyAverage = {}
with open(RESULTS_PATH + "/" + json_file_name, 'r') as f:
    for line in f:
        obj = json.loads(line)
        if dailyAverage[obj[timestamp]]:
            dailyAverage[f'{obj[timestamp]}'].append(obj['test_score'][0])
        else:
            dailyAverage[f'{obj[timestamp]}'] = [obj['test_score'][0]]
print(dailyAverage)


