import snscrape.modules.twitter as sntwitter
import pandas as pd

# query = "(from:elonmusk) until:2022-11-27 since:2019-01-01"
query = "(from:elonmusk) until:2022-11-27 since:2019-01-01 -filter:replies"
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