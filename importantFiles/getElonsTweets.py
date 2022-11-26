# For sending GET requests from the API
import requests
# For saving access tokens and for file management when creating and adding to the dataset
import os
# For dealing with json responses we receive from the API
import json
# For displaying the data after
import pandas as pd
# For saving the response data in CSV format
import csv
# For parsing the dates received from twitter in readable formats
import datetime
import dateutil.parser
import unicodedata
#To add wait time between requests
import time
# Importing the libraries
import configparser
import tweepy

# Read the config file
config = configparser.ConfigParser()
config.read('5Y Masters\Machine Learning\MachineLearning-GroupProject\importantFiles\config.ini')

# Read the values
api_key = config['twitter']['api_key']
api_key_secret = config['twitter']['api_key_secret']
access_token = config['twitter']['access_token']
access_token_secret = config['twitter']['access_token_secret']

# Authenticate
auth = tweepy.OAuthHandler(api_key, api_key_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth)

user = 'elonmusk'
limit = 100
# tweets = tweepy.Cursor(api.user_timeline, screen_name = user, count = 200, tweet_mode = 'extended', exclude_replies=True, include_rts=False,).items(limit)
page_no = 1
pages = tweepy.Cursor(api.user_timeline, screen_name=user, tweet_mode='extended').pages(limit)

columns = ['Date','Tweet']
data = []

# Iterate through the pages and print the text of the tweets
for page in pages:    
    for i in range(len(page)):
        # print(f"{i+1} {page[i].created_at, page[i].full_text } \n")
        data.append([page[i].created_at, page[i].full_text])
    print(f"*********End of Page {page_no}*********")
    page_no += 1


df = pd.DataFrame(data, columns=columns)

print(df)
# df.to_csv("dummy.csv")

df.to_csv("5Y Masters\Machine Learning\MachineLearning-GroupProject\importantFiles\ElonsAlltweets.csv", encoding='utf_8_sig')



# with open("dummy.csv", errors='replace') as fin:
#     with open('ElonsAlltweets.csv', 'w', encoding='cp932') as f_out:
#         f_out.write(fin.read())
    # read_data = fin.read()
    # print(read_data)
    # aaa = pd.DataFrame(fin)
    # aaa.to_csv("aaa.csv")











# class Listener(tweepy.Stream):

#     tweets = []
#     limit = 10

#     def on_status(self, status):
#         self.tweets.append(status)

#         if len(self.tweets) == self.limit:
#             self.disconnect()


# stream_tweet = Listener(api_key, api_key_secret, access_token, access_token_secret)
# user_id = api.get_user(screen_name = user).id
# print(user_id)

# stream_tweet.filter(follow = str(user_id))


# for tweet in tweets:
#     data.append([tweet.user.screen_name, tweet.full_text])
    # print(tweet.full_text)

# df = pd.DataFrame(data, columns=columns)

# for tweet in public_tweets:
#     print(tweet.text)
#     print(tweet.created_at)
#     print(tweet.id)
#     print(tweet.user.screen_name)
#     print(tweet.user.name)
#     print(tweet.user.followers_count)


# Output: 'QuantInsti'


# using get_user with screen_name
# put_your_screen_name = "quantinsti"
# user2 = api.get_user(screen_name=put_your_screen_name)
# user2.id
# Output: 869660137