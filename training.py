from cProfile import label
from re import X
# from statistics import LinearRegression
from tkinter import Y
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

df = pd.read_csv('/Users/doimasanari/Desktop/datasetMasanariDoi.csv')

x = df.iloc[:,1]  # construct a matrix coniaing tweets
y = df.iloc[:,2]  # construct a matrix coniaing -1, 0 or 1

xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=0.2) # split the data for training and testing.
xTest = np.array(xTest) #make an array of x test data


def logisticRegression(xTrain, yTrain): # train data by logistic Regression
    model = LogisticRegression()
    model.fit(xTrain, yTrain)                       # train data
    print("slope = ", model.coef_)                       # get a slope here
    print("intercept = ", model.intercept_)              # get an intercept here
    print("train score = ", format(model.score(xTrain, yTrain)))
    predData = np.array(model.predict(xTest))   
    predData = predData.reshape(-1,1)            # make a tidy array of prediction data which contains values, -1, 0 or 1


def linear_SVC (c, xTrain, yTrain):     # train data by SVC
    model = LinearSVC(C=c).fit(xTrain, yTrain)                         # train a data
    print("when C =", c)                                                    
    print("slope = ", model.coef_)                                          # get a slope here
    print("intercept = ", model.intercept_)                                 # get an intercept here
    predData = np.array(model.predict(xTest))   
    predData = predData.reshape(-1,1)            # make a tidy array of prediction data which contains values, -1, 0 or 1


logisticRegression(xTrain, yTrain)
linear_SVC (0, xTrain, yTrain)
linear_SVC (0.001, xTrain, yTrain)
linear_SVC (10, xTrain, yTrain)
linear_SVC (100, xTrain, yTrain)