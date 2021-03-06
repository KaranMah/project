# Financial Data Forecaster

Repository for final year project of building a financial data forecaster.

Need to add more description crap here

## How to Run the Code:

### Data Collection

In order to collect the data, the relevant data files have to changed to select the countries, currencies, stock exchanges and heads of state for which data is to be collected.

* currencies.txt
* cur_pairs.txt
* social_media.txt

The collection of data happens with the several scripts mentioned below to collect financial data, along with data from Twitter and Reddit.

#### Financial Data Collection 

Scripts for collecting financial data. It can be run as follows.

```shell
python num_data.py
```

#### Social Media Data Collection


The script to build the text files containing the data queries. It can be run as follows.

```shell
python text_data.py
```

##### Twitter Data

These scripts are to run iteratively to collect data as long as there are incomplete queries to be scraped. The script run in the background as follows:

```shell
python run_twitter_query.py
python move_scrapy_results.py
python update_twitter_query.py
```

These scripts convert the Twitter data to the relevant format for BERT processing.

```shell
./data_to_json.sh
./json_to_bert-csv.sh
```

##### Reddit Data

Scripts to collect the Reddit data.

```shell
python run_reddit_query.py
python move_reddit_results.py
```

Script to convert Reddit results for BERT processing.

```shell
python reddit_json_csv.py
```

### Sentiment Analysis

Script to use BERT on collected posts from social media to obtain sentiment scores.

```shell
python test_bert.py
```

### Feature Engineering

Scripts to aggregate sentiment scores and build new features using financial data.

```shell
python feature_engineering_1.py
python create_sentiment_aggregate.py
```

### Optimized Modelling

The script to run the optimized selected model

```shell
python sec_opt.py
python optimization_mult.py
```

### Backtesting 

The script to run the backtesting strategy to evaluate performance.

```shell
python backtest_portfolio.py
python bcaktest_second.py
```

## Results Obtained

### Optimized Model Approach

Ridge Classifier classification on 80-20 split

![Ridge Classifier classification on 80-20 split](https://github.com/KaranMah/project/blob/master/Additional_docs/80-20%20split%20BDT%20classification.png?raw=true)

Accuracy plotted against various hyperparameters for Ridge Classifier
![accuracy plot](https://github.com/KaranMah/project/blob/master/Additional_docs/BDT%20Ridge%20Classsification%20optimization%20acuracy%20plot.png?raw=true)

Walk forward forecasting results for chosen markets compared with a regular 80-20 split 
![walk forward](https://github.com/KaranMah/project/blob/master/Additional_docs/Walk%20Forward%20forecasting%20results.png?raw=true)

Ensemble voting for best performing approaches: walk forward for forex and 80-20 split for Index
![ensemble](https://github.com/KaranMah/project/blob/master/Additional_docs/Ensemble%20voting%20results.png?raw=true)

The above results highlighted the best approaches to get most accurate predictions for the chosen markets using SVM and Ridge Classifier.

### Backtesting 
Performance of two backtesting models: moving average and intraday trading based on predictions were analysed. Both strategies showed significant gains in equity. The results for BDT and MNT markets are shown below. Testing period for these models was Jan 2018-Dec 2019.

Moving average backtesting
![moving average backtesting](https://github.com/KaranMah/project/blob/master/Additional_docs/optimized%20moving%20average%20backtesting%20.png?raw=true)

Intraday trading strategy
![intraday trading](https://github.com/KaranMah/project/blob/master/Additional_docs/ptimized%20intraday%20trading.png?raw=true)
