# Financial Data Forecaster - Repository

Repository for final year project of building a financial data forecaster.

Need to add more description crap here

## How to Run the Code:

### Data Collection

In order to collect the data, the relevant data files have to changed to select the countries, currencies, stock exchanges and heads of state for which data is to be collected.

* currencies.txt
* cur_pairs.txt
* social_media.txt

The collection of data happens with the several scripts mentioned below to collect financial data, along with data from Twitter and Reddit.

### Financial Data Collection 

Scripts for collecting financial data:

* num_data.py

It can be run as 
```shell
python num_data.py
```

### Social Media Data Collection

* text_data.py

#### Twitter Data

1. run_twitter_query.py
2. move_scrapy_results.py
3. update_twitter_queries.py

These scripts are to run iteratively to collect data as long as there are incomplete queries to be scraped.



