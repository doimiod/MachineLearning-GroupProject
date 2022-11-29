import snscrape.modules.twitter as sntwitter
import numpy as np
import pandas as pd
#import external pandas_datareader library with alias of web

Address = "5Y Masters\Machine Learning\MachineLearning-GroupProject\importantFiles/"
# query = "(from:elonmusk) until:2022-11-27 since:2019-01-01"
query = "(from:elonmusk) since:2010-06-29 -filter:replies"
tweets = []
limit = 100000000000


i=0
for tweet in sntwitter.TwitterSearchScraper(query).get_items():

    if(i%100 == 0):
        print("progress: {} tweets".format(i))
        print("lenght of tweets: {} \n".format(len(tweets)))
    i = i+1
    # print(vars(tweet))
    # break
    try:
        if len(tweets) == limit:
            break
        else:
            tweets.append([tweet.date, tweet.username, tweet.content])
    except:
        print("some weird error")
np.flip(tweets,0)
df = pd.DataFrame(tweets, columns=['Date', 'User', 'Tweet'])

print(df)

df.to_csv("{}ElonsTweets.csv".format(Address), encoding='utf_8_sig')
E_date = df['Date']
E_size = len(E_date)

import yfinance as yf
start_date = E_date[E_size-1]
end_date = E_date[1]
ticker = 'TSLA'
data = yf.download(ticker, start_date, end_date)
data["Date"] = data.index

data = data[["Date", "Open", "High",
"Low", "Close", "Adj Close", "Volume"]]

data.reset_index(drop=True, inplace=True)
print(data.head())
data.to_csv("{}TSLA.csv".format(Address))