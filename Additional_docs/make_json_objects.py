import sys
import json

fileName = sys.argv[1]
json_file = "/data/json/"+fileName.split('/')[-1]+".json" 
print(json_file)
num_lines = sum(1 for line in open(fileName))
f1 = open(json_file, 'a')
f1.write('[')
c = 2
print(num_lines)
with open(fileName, 'r') as f2:
    line = f2.readline()
    f1.write(line.strip('\n'))
    while(line):
        line = f2.readline()
        f1.write(line.strip('\n'))
        if(c < num_lines):
            f1.write(',')
        c = c + 1
print("Done writing for "+fileName+"...")
f1.write(']')
f1.close()
