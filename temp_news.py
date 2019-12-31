import re
import json
import requests

location = "hongkong"
start = "2019-01-01"
end = "2019-10-01"
gnews_token = "9f1b37c04efb82e0854da454f9ad00b3"
url = "https://gnews.io/api/v3/search?q="+location+"&mindate="+start+"&maxdate="+end+"&token="+gnews_token



news_api_token = "d3f8658b28c541abb484d3bb19ed856a"

url_v2 = "https://newsapi.org/v2/everything?q="+location+"&from="+start+"&to="+end+"&apiKey="+news_api_token

r = requests.get(url_v2)
r.text
