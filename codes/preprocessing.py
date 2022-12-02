from ast import Add
import numpy as np
import pandas as pd
import datetime
import time
from dateutil import parser
import matplotlib.pyplot as plt
import snscrape.modules.twitter as sntwitter

# #End with slash / if using directory or leave blank ""
Address = "importantFiles/"

# #CAREFUL only for deprecation of datetime64
import warnings
warnings.filterwarnings("ignore")
# #CAREFUL

#Initialization
threshold = 0.02
Elon_class_avail = True
Elon_data_avail = True
update_tweet = False
update_class = False
Elon_class = []
Elon_data = []
try:
    Elon_data = pd.read_csv("{}ElonsTweets1.csv".format(Address))
except:
    Elon_data_avail = False
try:
    E_date = Elon_data['Date']
    E_date = np.array(E_date, dtype=np.datetime64)
    E_date = E_date.astype(datetime.datetime)
    E_date = [x.strftime('%Y-%m-%d') for x in E_date]
    E_size = len(E_date)
except: 
    x=1
try:
    Elon_class = pd.read_csv("{}Elon_class1.csv".format(Address))
except:
    Elon_avail = False

if(Elon_data_avail):
    val = input("Elon's tweets are found, update it? [y/n]")
    if (val == 'y'):
        update_tweet = True

#________________GATHERING___TWEETS_______________________________
if(not Elon_data_avail or update_tweet):
    print(" Gathering Tweets...")
    query = "(from:elonmusk) since:2010-06-29"
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
    tweets = np.flip(tweets,0)
    df = pd.DataFrame(tweets, columns=['Date', 'User', 'Tweet'])

    print(df)

    df.to_csv("{}ElonsTweets1.csv".format(Address), encoding='utf_8_sig')
    E_date = df['Date']
    E_size = len(E_date)
else:
    if(update_tweet):
        print("tweets updated")

#_________________________creating_data_for_training____________________

val = input("Elon's classes are found, update it? [y/n]")
if (val == 'y'):
    update_class = True

#starting point
#class increases: 2
#class decreses: -2
#class undecided: 0
#classes for stocks between dates. need to assign the class to the tweets at the relevant dates.

#
if(not Elon_class_avail or update_class):
    Elon_data = pd.read_csv("{}ElonsTweets1.csv".format(Address))
    #convert dict to array to datetime64 to only yyyy-mm-hh
    E_date = Elon_data['Date']
    E_date = np.array(E_date, dtype=np.datetime64)
    E_date = E_date.astype(datetime.datetime)
    E_date = [x.strftime('%Y-%m-%d') for x in E_date]
    E_size = len(E_date)

    # initial_date = E_date[E_size-1]
    Elon_class = []
    for i in range(0, E_size):
        Elon_class.append([E_date[i], Elon_data['Tweet'][i],"",""])

    import yfinance as yf

    class_found=False
    class_val = 0
    ticker = 'TSLA'
    
    for i in range(0,E_size):
        #find class
        count=i
        while(not class_found):
            #find index of next date
            j=0

            try:
                while(E_date[count+j]<=E_date[i]):
                    j=j+1
            except:
                break

            #download tsla stocks for the 2 dates
            data = yf.download(ticker, E_date[count], E_date[count+j])
            data["Date"] = data.index
            size = len(data)

            #try to find a valid close price, if less than 2 dates found, decrement count
            try:
                if(data['Close'][size-1]-data['Close'][0] > data['Close'][0]*threshold):
                    class_val = 2
                elif(data['Close'][size-1]-data['Close'][0] < -data['Close'][0]*threshold):
                    class_val = -2
                else:
                    class_val = 0
                class_found= True
            except:
                count = count - 1

        #assign class
        Elon_class[i][2] = class_val
        Elon_class[i][3] = data['Close'][0]
        #if a new date is next, set class_found to false, it may throw error if we go out of bounds
        try:
            if(E_date[i+1]>E_date[i]):
                class_found = False
                print("Processing date: {}".format(E_date[i+1]))
        except:
            x=1

    #store Elon_data
    columns = ['Date','Tweet', 'Class', 'Close_price']
    df = pd.DataFrame(Elon_class, columns=columns)
    df.to_csv("{}Elon_class1.csv".format(Address), encoding='utf_8_sig')
else:
    if(Elon_class_avail):
        print("class available")
    else:
        print("class unavailable")
    for i in range(0, len(Elon_class)):
        try:
            j=0
            
            while(E_date[i+j]<=E_date[i]):
                j=j+1
        except:
            Elon_class['Date'][i] = 0
            break

        if(Elon_class['Close_price'][i+j]-Elon_class['Close_price'][i] > Elon_class['Close_price'][i]*threshold):
            class_val = 2
        elif(Elon_class['Close_price'][i+j]-Elon_class['Close_price'][i] < -Elon_class['Close_price'][i]*threshold):
            class_val = -2
        else:
            class_val = 0
        Elon_class['Class'][i] = class_val
    Elon_class.to_csv("{}Elon_class1.csv".format(Address), encoding='utf_8_sig')

Elon_class = pd.read_csv("{}Elon_class1.csv".format(Address))

#remove URL
import re

for i in range(0,len(Elon_class)):
    try:
        Elon_class['Tweet'][i] = re.sub(r'http\S+', '', Elon_class['Tweet'][i])
        Elon_class['Tweet'][i] = re.sub(r'@', '', Elon_class['Tweet'][i])
    except:
        x=1

Elon_class.to_csv("{}Elon_class1.csv".format(Address), encoding='utf_8_sig',)

#sum of individual classes
class1 = 0
class1p = 0
class_n = 0
for i in range(0,len(Elon_class)-1):
    if(Elon_class['Class'][i] == 2):
        class1 = class1+1
    if(Elon_class['Class'][i] == -2):
        class1p = class1p+1
    if(Elon_class['Class'][i] == 0):
        class_n = class_n+1

print("amt of 1: {}, amt of -1: {}, amt of 0: {}".format(class1, class1p, class_n))