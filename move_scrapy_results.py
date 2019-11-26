import re
import os
import sys
import argparse
import subprocess
from datetime import datetime, timedelta

parser = argparse.ArgumentParser(description='Script to move Scrapy results')
parser.add_argument('--source', action="store", default="TweetScraper-master/Data/tweet", dest="source", help="Enter directory of files")
parser.add_argument('--destination', action="store", default="TweetScraper-master", dest="dest", help="Enter directory to move files")
source = parser.parse_args().source
dest = parser.parse_args().dest

files = os.popen('ls '+dest+' | grep -E "nohup"').read()
queries = [x[6:-4] for x in files]
