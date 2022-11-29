from ast import Add
import numpy as np
import pandas as pd
import time
from dateutil import parser
import matplotlib.pyplot as plt

# #End with slash / if using directory or leave blank ""
Address = "5Y Masters\Machine Learning\MachineLearning-GroupProject\importantFiles/"

# #CAREFUL only for deprecation of datetime64
import warnings
warnings.filterwarnings("ignore")
# #CAREFUL

Elon_data = pd.read_csv("{}ElonsTweets.csv".format(Address))
stock_data  = pd.read_csv("{}TSLA.csv".format(Address))

# print(stock_data['Date'])

E_date = Elon_data['Date']
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

for i in range(1,S_size):
    #stock increases
    if(stock_data['Close'][i]-stock_data['Close'][i-1] > stock_data['Close'][i-1]*0.05):
        Class.append([stock_data['Date'][i-1], 1])
    #stock decreases
    elif(stock_data['Close'][i]-stock_data['Close'][i-1] < -stock_data['Close'][i-1]*0.05 ):
        Class.append([stock_data['Date'][i-1], -1])
    #undecided
    else:
        Class.append([stock_data['Date'][i-1], 0])
    print("the day {}, previous day {}, class {}".format(stock_data['Close'][i],stock_data['Close'][i-1],Class[i-1]))

initial_date = E_date[E_size-1]

Elon_class = []
for i in range(0, E_size):
    Elon_class.append([])

for i in range(1,S_size):
    for j in range(0,E_size):
        #exit if all tweets for the date is assigned
        if(E_date[j]>S_date[i]):
            break
        #skip if the for loop hasnt reached the stock date.
        


#sum of individual classes
class1 = 0
class1p = 0
class_n = 0
for i in range(0,S_size-1):
    if(Class[i][1] == 1):
        class1 = class1+1
    if(Class[i][1] == -1):
        class1p = class1p+1
    if(Class[i][1] == 0):
        class_n = class_n+1

print("amt of 1: {}, amt of -1: {}, amt of 0: {}".format(class1, class1p, class_n))

#store classes
columns = ['Date', 'Class']
df = pd.DataFrame(Class, columns=columns)
df.to_csv("{}Classes.csv".format(Address), encoding='utf_8_sig')



