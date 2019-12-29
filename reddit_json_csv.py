import os
import re
import ast
import json
import shlex
import subprocess
from datetime import datetime, time

file_out = os.popen('ls . | grep -E "^r_.*"')
files = file_out.read()
files = files.strip('\n').split('\n')


for file in files:
    with open(file, 'rb') as f:
        line = f.readline().decode()
        data = ast.literal_eval(line)
        print(data)
        lol
