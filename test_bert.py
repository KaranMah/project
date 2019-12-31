import os
import json
from os import listdir
from os.path import isfile, join
import ast
import pandas as pd

BERT_DIR = "/home/fyp19020/BERT-fine-tuning-for-twitter-sentiment-analysis"
BERT_DATA_DIR = f'{BERT_DIR}/data'
BERT_MODEL = "MODEL NAME"

csvPath = '/data/csv/'
allFiles = [f for f in listdir(csvPath) if isfile(join(csvPath, f))]

# get first item that isn't processed
# next(x for x in allFiles if x)

print(allFiles)
print("\n")
file_name = allFiles[0]
print(file_name)

TASK_NAME = file_name
OUTPUT_DIR = f'{BERT_DATA_DIR}/outputs/{TASK_NAME}/'
REPORTS_DIR = f'{BERT_DATA_DIR}/reports/{TASK_NAME}_evaluation_reports/'

#get data
print("reading file\n")
with open(csvPath + file_name, 'r') as f:
    raw_input = f.readlines()
print("file read\n")


# Convert data to input file
lst = [x.replace('\n', '') for x in raw_input]
df = pd.DataFrame(lst)
df.to_csv(BERT_DIR + '/data/get_test.csv', sep='\t', index=False, header=False)
print("input ready...running bert\n")
print(df.shape)

# Run Command
if os.path.exists(f'{BERT_DATA_DIR}/bert_result/test_results.tsv'):
        os.remove(f'{BERT_DATA_DIR}/bert_result/test_results.tsv')

COMMAND = f'python3 {BERT_DIR}/test_bert/run_classifier.py \
     --task_name=twitter \
     --do_predict=true \
     --data_dir={BERT_DIR}/data \
     --vocab_file={BERT_DIR}/Bert_base_dir/vocab.txt\
     --bert_config_file={BERT_DIR}/Bert_base_dir/bert_config.json\
     --init_checkpoint={BERT_DIR}/model \
     --max_seq_length=64 \
     --output_dir={BERT_DIR}/data/bert_result'
os.system(COMMAND)

with open(f'{BERT_DATA_DIR}/bert_result/test_results.tsv', 'r') as f:
    test_results = f.readlines()

print("BERT successful\n")
# Append data to input
jsonPath = "/data/json"

json_file_name = file_name.replace(".csv", ".json")

obj = []
i=0
with open(jsonPath + "/" + json_file_name, 'rb') as f:
    line = f.readline().decode()
    while line:
        data = ast.literal_eval(line)
        data['test_score'] = test_results[i]
        obj.append(data)
        line = f.readline().decode()

print("saving results")
# saving data
resultsPath = "/data/results"

with open(resultsPath +"/"+ json_file_name, 'w') as f:
    for item in obj:
        f.write(json.dumps(item))
        f.write("\n")

