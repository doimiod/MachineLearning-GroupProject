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
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
nltk.download('punkt')
nltk.download('stopwords')
from sklearn.feature_extraction.text import TfidfVectorizer
from  sklearn.metrics  import accuracy_score

# df = pd.read_csv('importantFiles\Elon_class.csv')
df = pd.read_csv('/Users/doimasanari/Desktop/MachineLearning-GroupProject/importantFiles/Elon_class.csv')

x = df['Tweet']  # construct a matrix containing tweets
y = df['Class']  # construct a matrix containing -1, 0 or 1

Tweets = df['Tweet'].str.cat(sep=' ')
#function to split text into word
tokens = word_tokenize(Tweets)
vocabulary = set(tokens)
print(len(vocabulary))
frequency_dist = nltk.FreqDist(tokens)
print(sorted(frequency_dist,key=frequency_dist.__getitem__, reverse=True)[0:50])
stop_words = set(stopwords.words('english'))
tokens = [w for w in tokens if not w in stop_words]
# print(tokens)
# text featuring ends
print(x)

xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=0.2) # split the data for training and testing.
print(type(xTrain))
xTest = np.array(xTest) #make an array of x test data

print(xTrain)

vectorizer = TfidfVectorizer()
# x = v.fit_transform(df['Review'].values.astype('U'))
xTrain = vectorizer.fit_transform(xTrain.astype('U').values)
xTest = vectorizer.transform(xTest.astype('U'))

def logisticRegression(xTrain, yTrain, xTest, yTest): # train data by logistic Regression
    model = LogisticRegression()
    model.fit(xTrain, yTrain)                       # train data
    print("slope = ", model.coef_)                       # get a slope here
    print("intercept = ", model.intercept_)              # get an intercept here
    print("train score = ", format(model.score(xTrain, yTrain)))
    print('\n')
    predData = np.array(model.predict(xTest))   
    predData = predData.reshape(-1,1)            # make a tidy array of prediction data which contains values, -1, 0 or 1
    print(accuracy_score(yTest,predData))


def linear_SVC (c, xTrain, yTrain):     # train data by SVC
    model = LinearSVC(C=c).fit(xTrain, yTrain)                         # train a data
    print("when C =", c)                                                    
    print("slope = ", model.coef_)                                          # get a slope here
    print("intercept = ", model.intercept_)                                 # get an intercept here
    print("mean accuracy = ", format(model.score(xTrain, yTrain)))
    print('\n')
    predData = np.array(model.predict(xTest))   
    predData = predData.reshape(-1,1)            # make a tidy array of prediction data which contains values, -1, 0 or 1


logisticRegression(xTrain, yTrain)
# linear_SVC (0, xTrain, yTrain)
linear_SVC (0.001, xTrain, yTrain)
linear_SVC (10, xTrain, yTrain)
linear_SVC (100, xTrain, yTrain)