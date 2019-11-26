import re
import os
import sys
import argparse
import subprocess
from datetime import datetime, timedelta

parser = argparse.ArgumentParser(description='Script to move Scrapy results')
parser.add_argument('--source', action="store", default="TweetScraper-master/Data/tweet", dest="source", help="Enter directory of files")
parser.add_argument('--dest', action="store", default="TweetScraper-master", dest="dest", help="Enter directory containing nohup files")
source = parser.parse_args().source
dest = parser.parse_args().dest
print(dest)
fileobj = os.popen('ls '+dest+' | grep -E "nohup"')
files = fileobj.read()
print(files)
queries = [x[5:-4] for x in files.strip('\n').split('\n')]
print(queries)
