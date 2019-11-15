from newsapi import NewsApiClient

newsapi = NewsApiClient(api_key='d3f8658b28c541abb484d3bb19ed856a')

everything = newsapi.get_everything(q='hongkong',
                                          from_param='2010-10-25',
                                          to='2010-10-27',
                                          language='en')
everything
