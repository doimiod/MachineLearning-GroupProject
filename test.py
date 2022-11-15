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

# using get_user with id
# put_your_user_id = "elonmusk"
# user1 = api.get_user(user_id = put_your_user_id)
# print(user1.name)

public_tweets = api.home_timeline()

for tweet in public_tweets:
    print(tweet.text)
    print(tweet.created_at)
    print(tweet.id)
    print(tweet.user.screen_name)
    print(tweet.user.name)
    print(tweet.user.followers_count)







# Output: 'QuantInsti'


# using get_user with screen_name
# put_your_screen_name = "quantinsti"
# user2 = api.get_user(screen_name=put_your_screen_name)
# user2.id
# Output: 869660137