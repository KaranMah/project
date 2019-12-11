import re
import os
import sys
import shlex
import argparse
import subprocess
from datetime import datetime, timedelta


parser = argparse.ArgumentParser(description='Script to run Reddit queries')
parser.add_argument('--log', action="store", default="Reddit_data/log/", dest="log", help="Enter log destination")
parser.add_argument('--data', action="store", default="Reddit_data/subreddit/", dest="data", help="Enter files destination")

log = parser.parse_args().log
data = parser.parse_args().data


print("Log files located in "+log+"...")
print("Data files located in "+data+"...")

fileobj = os.popen('ls '+log+' | grep -E "nohup"')
files = fileobj.read()
files_array = files.strip('\n').split('\n')
nohup_files = list(filter(lambda x: "nohup" in x, files_array))
for file in nohup_files:
    query = file[5:-4]
    print("Subreddit: "+query)
    print("Checking if scraping results for "+query+" are completed...")
    out = os.popen('tail '+log+'/'+file+' -n 1')
    last = out.read()
    if("Completed" in last):
        print("Scraping is completed")
        print("Locating data file for "+query+"...")
        if(os.path.exists(log+query)):
            print("Data file exists for "+query+"...")
            from_date = "01-01-2010"
            to_date = "01-01-2019"
            print("Data file for "+query+" has scraped from "+from_date+" to "+to_date)
            file_name = 'r/'+query+'_'+from_date+'_'+to_date
            os.popen('mv '+data+query+' /data/'+file_name)
            print("Moved data file...")
            os.popen('rm '+log+file)
            print("Removed nohup file...")
            print()
