import snscrape.modules.twitter as sntwitter
import pandas as pd
#import external pandas_datareader library with alias of web

# query = "(from:elonmusk) until:2022-11-27 since:2019-01-01"
query = "(from:elonmusk) since:2019-1-01 -filter:replies"
tweets = []
limit = 100000000000


for tweet in sntwitter.TwitterSearchScraper(query).get_items():
    
    # print(vars(tweet))
    # break
    if len(tweets) == limit:
        break
    else:
        tweets.append([tweet.date, tweet.username, tweet.content])
        
df = pd.DataFrame(tweets, columns=['Date', 'User', 'Tweet'])

print(df)

df.to_csv("ElonsTweets.csv", encoding='utf_8_sig')
 
import yfinance as yf
start_date = '2019-01-01'
end_date = '2022-11-28'
ticker = 'TSLA'
data = yf.download(ticker, start_date, end_date)
data["Date"] = data.index

data = data[["Date", "Open", "High",
"Low", "Close", "Adj Close", "Volume"]]

data.reset_index(drop=True, inplace=True)
print(data.head())
data.to_csv("TSLA.csv")