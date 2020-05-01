import os
import sys
import json

fileName = sys.argv[1]
data_file = "/data/json/"+fileName.split('/')[-1]+".json"
print(data_file)

fw = open(data_file, 'a')
fw.write('[')

with open(fileName, 'r') as fr:
    while(True):
        c = fr.read(1)
        if(not c):
            break
        elif(c == '}'):
            fw.write(c)
            fw.write(',')
        else:
            fw.write(c)
fw.close()

os.system("truncate -s-1  "+data_file)
os.system("echo ']' >> "+data_file)
