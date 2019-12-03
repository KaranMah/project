import os
import re
import json

file_out = os.popen('ls . | grep -E "#.*"')
files = file_out.read()
files = files.strip('\n').split('\n')
print(files)

for file_name in files:
    if(os.path.exists('/data/csv/'+file_name+'.csv')):
        pass
    else:
        os.popen('jq -r \'[.[]] | @csv\' '+file_name+' > /data/csv/'+file_name+'.csv')

