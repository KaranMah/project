import os
from os import listdir
from os.path import isfile, join
import sys
import pandas as pd

BERT_DIR = "/home/fyp19020/BERT-fine-tuning-for-twitter-sentiment-analysis"
BERT_DATA_DIR = f'{BERT_DIR}/data'
BERT_MODEL = "MODEL NAME"

csvPath = '/data/csv/'
allFiles = [f for f in listdir(csvPath) if isfile(join(csvPath, f))]

# get first item that isn't processed
# next(x for x in allFiles if x)

file_name = allFiles[0]
print(file_name)

TASK_NAME = file_name
OUTPUT_DIR = f'{BERT_DATA_DIR}/outputs/{TASK_NAME}/'
REPORTS_DIR = f'{BERT_DATA_DIR}/reports/{TASK_NAME}_evaluation_reports/'

#get data
with open(csvPath + file_name, 'r') as f:
    raw_input = f.readlines()

# Convert data to input file
lst = [x.replace('\n', '') for x in raw_input]
df = pd.DataFrae(lst)
df.to_csv(BERT_DIR + 'data/get_test.tsv', sep='\t', index=False, header=True)


# Run Command
os.system(f'cd {BERT_DIR}')
COMMAND = f'python ./test_bert/run_classifier.py \
     --task_name={TASK_NAME} \
     --do_predict=true \
     --data_dir=./data \
     --vocab_file=./Bert_base_dir/vocab.txt\
     --bert_config_file=./Bert_base_dir/bert_config.json\
     --init_checkpoint=./model \
     --max_seq_length=64 \
     --output_dir=./data/bert_result'
os.system(COMMAND)
with open(f'{BERT_DATA_DIR}/bert_results/test_results.tsv'):
    test_results = f.readlines()

# Append data to input
jsonPath = "/data/JSON"
with open(jsonPath + file_name, 'r') as f:
    json_input = f.readlines()

for i in range(len(json_input)):
    json_input[i]["test_score"] = test_results[i]

# saving data
resultsPath = "/data/results"
with open(resultsPath + file_name, 'w') as f:
    for obj in json_input:
        f.write(obj)








