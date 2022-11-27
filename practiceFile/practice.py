import tweepy
import datetime
import sys
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
import pytz
utc=pytz.UTC

# Read the config file
config = configparser.ConfigParser()
config.read('config.ini')

# Read the values
api_key = config['twitter']['api_key']
api_key_secret = config['twitter']['api_key_secret']
access_token = config['twitter']['access_token']
access_token_secret = config['twitter']['access_token_secret']

# Authenticate
auth = tweepy.OAuthHandler(api_key, api_key_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth)

username = 'elonmusk'
startDate = utc.localize(datetime.datetime(2019, 12, 1, 0, 0, 0))
endDate = utc.localize(datetime.datetime(2019, 12, 10, 0, 0, 0))

columns = ['Date','Tweet']
data = []

tmpTweets = api.user_timeline(username)

for tweet in tmpTweets:
    if tweet.created_at < endDate and tweet.created_at > startDate:
        data.append(tweet)

while (tmpTweets[-1].created_at > startDate):
    print("Last Tweet @", tmpTweets[-1].created_at, " - fetching some more")
    tmpTweets = api.user_timeline(username, max_id = tmpTweets[-1].id)
    for tweet in tmpTweets:
        if tweet.created_at < endDate and tweet.created_at > startDate:
            data.append([tweet.created_at, tweet])

df = pd.DataFrame(data, columns=columns)

print(df)
# df.to_csv("dummy.csv")

df.to_csv("ElonsAllTweets.csv", encoding='utf_8_sig')