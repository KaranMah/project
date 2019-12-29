import argparse
import datetime
from psaw import PushshiftAPI

api = PushshiftAPI()

parser = argparse.ArgumentParser(description='Reddit scraper')
parser.add_argument('--query', action="store", dest="subreddit", help="Enter file name", required=True)
parser.add_argument('--start', action="store", dest="start", help="Enter start date dd/mm/yyyy", required=True)
parser.add_argument('--end', action="store",  dest="end", help="Enter end date dd/mm/yyyy", required=True)
parser.add_argument('--data', action="store", default="Reddit_data/subreddit/", dest="data", help="Enter files destination")

subreddit = parser.parse_args().subreddit
start_date = parser.parse_args().start
end_date = parser.parse_args().end
data_folder = parser.parse_args().data

class RedditScraper(object):
    def __init__(self,  api, subreddit, earliest_date, latest_date, limit=None):
        self.earliest_date = (datetime.datetime.strptime(earliest_date, "%d/%m/%Y")).strftime("%d-%m-%Y")
        self.latest_date = (datetime.datetime.strptime(latest_date, "%d/%m/%Y")).strftime("%d-%m-%Y")
        self.limit = 100
        self.subreddit = subreddit
        self.api = api
        self.filter = ['url', 'author', 'title','submission', 'subreddit']

    def search_date(self, start_date=None, end_date=None):
        dates = []
        start_date = self.earliest_date if start_date is None else start_date
        end_date = self.latest_date if end_date is None else end_date
        start = datetime.datetime.strptime(start_date, "%d-%m-%Y")
        end = datetime.datetime.strptime(end_date, "%d-%m-%Y")
        delta = datetime.timedelta(days=1)
        while start <= end:
            dates.append(start)
            start += delta
        dates.append(start)
        return dates

    def scrape_subreddit(self,dates):
        data = []
        print("Scraping for query "+self.subreddit+" from "+self.earliest_date+" to "+self.latest_date)
        for i in range(len(dates)-1):
            if(i % 100 == 0):
                print(dates[i])
            record = api.search_submissions(after=int(dates[i].timestamp()), before=int(dates[i+1].timestamp()), subreddit=self.subreddit,filter=self.filter, limit=self.limit)
            if(record != []):
                data.extend([dict(obj.d_, timestamp=dates[i].strftime("%d-%m-%Y")) for obj in record])
        with open(data_folder+self.subreddit+"_"+self.earliest_date+"_"+self.latest_date, 'w', encoding='utf-8') as f:
            for item in data:
                f.write("%s\n" % item)
        print("Completed")


scraper = RedditScraper(api, subreddit, start_date, end_date)
print(scraper.subreddit, scraper.earliest_date, scraper.latest_date)
dates = scraper.search_date()
scraper.scrape_subreddit(dates)
