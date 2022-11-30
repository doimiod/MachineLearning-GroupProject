from ast import Add
import numpy as np
import pandas as pd
import datetime
import time
from dateutil import parser
import matplotlib.pyplot as plt

# #End with slash / if using directory or leave blank ""
Address = "importantFiles/"

# #CAREFUL only for deprecation of datetime64
import warnings
warnings.filterwarnings("ignore")
# #CAREFUL

Elon_data = pd.read_csv("{}ElonsTweets.csv".format(Address))
stock_data  = pd.read_csv("{}TSLA.csv".format(Address))

# print(stock_data['Date'])

E_date = Elon_data['Date']
E_date = np.array(E_date, dtype=np.datetime64)
E_date = E_date.astype(datetime.datetime)
E_date = [x.strftime('%Y-%m-%d') for x in E_date]
S_date = stock_data['Date']
E_size = len(E_date)
S_size = len(S_date)
Class = []


#dont delete the below code lol/ testing time arithemetic
# z = np.array(E_date[E_size-1], dtype=np.datetime64)
# print(z)

# print(z+np.timedelta64(5,'D'))
# d = z+np.timedelta64(5,'D')
# if(z+np.timedelta64(5,'D') == d):
#     print('works')

#starting point
#class increases: 1
#class decreses: -1
#class undecided: 0
#classes for stocks between dates. need to assign the class to the tweets at the relevant dates.

# for i in range(1,S_size):
#     #stock increases
#     if(stock_data['Close'][i]-stock_data['Close'][i-1] > stock_data['Close'][i-1]*0.05):
#         Class.append([stock_data['Date'][i-1], 1])
#     #stock decreases
#     elif(stock_data['Close'][i]-stock_data['Close'][i-1] < -stock_data['Close'][i-1]*0.05 ):
#         Class.append([stock_data['Date'][i-1], -1])
#     #undecided
#     else:
#         Class.append([stock_data['Date'][i-1], 0])
#     print("the day {}, previous day {}, class {}".format(stock_data['Close'][i],stock_data['Close'][i-1],Class[i-1]))

initial_date = E_date[E_size-1]
Elon_class = []
for i in range(0, E_size):
    Elon_class.append([E_date[i], Elon_data['Tweet'][i],""])

import yfinance as yf

class_found=False
class_val = 0
ticker = 'TSLA'
threshold = 0.05
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
                class_val = 1
            elif(data['Close'][size-1]-data['Close'][0] < -data['Close'][0]*threshold):
                class_val = -1
            else:
                class_val = 0
            class_found= True
        except:
            count = count - 1

    #assign class
    Elon_class[i][2] = class_val

    #if a new date is next, set class_found to false, it may throw error if we go out of bounds
    try:
        if(E_date[i+1]>E_date[i]):
            class_found = False
            print("Processing date: {}".format(E_date[i+1]))
    except:
        x=1
    
#store Elon_data
columns = ['Date','Tweet', 'Class']
df = pd.DataFrame(Elon_class, columns=columns)
df.to_csv("{}Elon_class.csv".format(Address), encoding='utf_8_sig')

#sum of individual classes
class1 = 0
class1p = 0
class_n = 0
for i in range(0,S_size-1):
    if(Elon_class[i][2] == 1):
        class1 = class1+1
    if(Elon_class[i][2] == -1):
        class1p = class1p+1
    if(Elon_class[i][2] == 0):
        class_n = class_n+1

print("amt of 1: {}, amt of -1: {}, amt of 0: {}".format(class1, class1p, class_n))

#store classes
columns = ['Date', 'Class']
df = pd.DataFrame(Class, columns=columns)
df.to_csv("{}Classes.csv".format(Address), encoding='utf_8_sig')



