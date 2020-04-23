import os
import json
from os import listdir
from os.path import isfile, join
import ast
import shutil

BERT_DIR = "/home/fyp19020/BERT-fine-tuning-for-twitter-sentiment-analysis"
BERT_DATA_DIR = f"{BERT_DIR}/data"
BERT_MODEL = "MODEL NAME"

csvPath = '/data/csv/'
allFilesNames = [f for f in listdir(csvPath) if isfile(join(csvPath, f))]
#countryList = ['Pakistan', 'Mongolia', 'Bangladesh', 'SriLanka', 'Karachi', 'Dhaka', 'Ulaanbaatar', 'Colombo']
countryList = ["#Ulaanbaatar"]
allFiles = [f for f in allFilesNames if any(j in f for j in countryList)]

file_name = ""
# get first item that isn't processed
for file in allFiles:
    file = file.replace(".csv", "")
    if not os.path.exists(f'/data/results/{file}.json'):
        file_name = file + ".csv"
        print(file_name)
        if file_name[0] == 'r':
            isTwitterFile = False
        else:
            isTwitterFile = True

        #####################
        dst_dir = f'{BERT_DATA_DIR}'
        src_file = csvPath + file_name
        shutil.copy(src_file, dst_dir)

        dst_file = os.path.join(dst_dir, file_name)
        new_dst_file_name = os.path.join(dst_dir, "get_test.csv")
        os.rename(dst_file, new_dst_file_name)

        # Run Command
        if os.path.exists(f'{BERT_DATA_DIR}/bert_result/test_results.tsv'):
            os.remove(f'{BERT_DATA_DIR}/bert_result/test_results.tsv')

        COMMAND = f'nohup python3 {BERT_DIR}/test_bert/run_classifier.py \
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
        #######################33

        jsonPath = "/data/json"

        json_file_name = file_name.replace(".csv", ".json")

        obj = []
        i = 0
        if not isTwitterFile:
            with open(jsonPath + "/" + json_file_name, 'rb') as f:
                line = f.readline().decode()
                while line:
                    data = ast.literal_eval(line)
                    test_results[i] = test_results[i][:-1]
                    data['test_score'] = test_results[i].split('\t')
                    obj.append(data)
                    line = f.readline().decode()
                    i += 1
        else:
            with open(jsonPath + "/" + json_file_name, 'r') as f:
                data = json.loads(f.read())

            for elem in data:
                test_results[i] = test_results[i][:-1]
                elem['test_score'] = test_results[i].split('\t')
                obj.append(elem)
                i += 1

        print("saving results in .... " + json_file_name)
        # saving data
        resultsPath = "/data/results"

        if os.path.exists(resultsPath + "/" + json_file_name):
            os.remove(resultsPath + "/" + json_file_name)

        with open(resultsPath + "/" + json_file_name, 'w+') as f:
            for item in obj:
                f.write(json.dumps(item))
                f.write("\n")

