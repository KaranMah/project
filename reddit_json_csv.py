import os
import re
import ast
import glob
import json
import shlex
import subprocess
from datetime import datetime, time

file_out = os.popen('ls /data | grep -E "^r_.*"')
files = file_out.read()
files = files.strip('\n').split('\n')

def write_to_json(file, data, year):
    with open("/data/json/"+file+"?"+str(year)+".json", 'w') as f1:
        for item in data:
            f1.write(json.dumps(item))
            f1.write("\n")

def json_to_csv(file, data, year):
    with open("/data/csv/"+file+"?"+str(year)+".csv", 'w') as f2:
        for item in data:
            f2.write(item['title'])
            f2.write("\n")


for file in files:
    obj = []
    if(len(glob.glob("/data/json/"+file+"*"))>0):
        print(file+" is done.")
        pass
    else:
        print("Need to move "+file)
        with open(file, 'rb') as f:
            line = f.readline().decode()
            while(line):
                data = ast.literal_eval(line)
                if(len(obj)==0):
                    obj.append(data)
                else:
                    if(datetime.strptime(data['timestamp'], "%d-%m-%Y").year == datetime.strptime(obj[0]['timestamp'], "%d-%m-%Y").year):
                        obj.append(data)
                    else:
                        json_to_csv(file, obj, datetime.strptime(obj[0]['timestamp'], "%d-%m-%Y").year)
                        write_to_json(file, obj, datetime.strptime(obj[0]['timestamp'], "%d-%m-%Y").year)
                        obj = [data]
                line = f.readline().decode()
            json_to_csv(file, obj, datetime.strptime(obj[0]['timestamp'], "%d-%m-%Y").year)
            write_to_json(file, obj,  datetime.strptime(obj[0]['timestamp'], "%d-%m-%Y").year)
