import datetime
from psaw import PushshiftAPI

api = PushshiftAPI()

class scrape_reddit(object):
    def __init__(self, limit=None, earliest_date=None, latest_date=None):
        self.earliest_date = '25/10/2019'
        self.latest_date = '26/10/2019'
        # self.latest_date = datetime.datetime.now().strftime("%d/%m/%Y")
        self.limit = 100
        self.subreddit = None
        self.filter = ['url', 'author', 'title', 'subreddit']

    def search_date(self, start_date=None, end_date=None):
        dates = []
        start_date = self.earliest_date if start_date is None else start_date
        end_date = self.latest_date if end_date is None else end_date
        start = datetime.datetime.strptime(start_date, "%d/%m/%Y")
        end = datetime.datetime.strptime(end_date, "%d/%m/%Y")
        delta = datetime.timedelta(days=1)
        while start <= end:
            dates.append(start)
            start += delta
        return dates

    def scrape_subreddit(self,dates):
        data = []
        for i in range(len(dates)-1):
            data.append(list(api.search_submissions(after=int(dates[i].timestamp()), before=int(dates[i+1].timestamp()), subreddit='hongkong',filter=self.filter, limit=self.limit)))
        data.append(list(api.search_submissions(after=int(dates[i].timestamp()), subreddit='hongkong',filter=self.filter, limit=self.limit)))
        with open('reddit_query.txt', 'w', encoding='utf-8') as f:
            for item in data:
                f.write("%s\n" % item)


scraper = scrape_reddit()
dates = scraper.search_date()
scraper.scrape_subreddit(dates)
