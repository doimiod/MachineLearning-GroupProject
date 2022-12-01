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
from sklearn.metrics import confusion_matrix, classification_report
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
nltk.download('punkt')
nltk.download('stopwords')
from sklearn.feature_extraction.text import TfidfVectorizer
from  sklearn.metrics  import accuracy_score

# #CAREFUL only for deprecation of datetime64
import warnings
warnings.filterwarnings("ignore")
# #CAREFUL

df = pd.read_csv('importantFiles\Elon_class.csv')
# df = pd.read_csv('/Users/doimasanari/Desktop/MachineLearning-GroupProject/importantFiles/Elon_class.csv')

x = df['Tweet']  # construct a matrix containing tweets
y = df['Class']  # construct a matrix containing -1, 0 or 1
date = df['Date'] # construct a matrix containing the date

Tweets = df['Tweet'].str.cat(sep=' ')
#function to split text into word
tokens = word_tokenize(Tweets)
vocabulary = set(tokens)
print(len(vocabulary))
frequency_dist = nltk.FreqDist(tokens)
print(sorted(frequency_dist,key=frequency_dist.__getitem__, reverse=True)[0:50])
stop_words = set(stopwords.words('english'))
tokens = [w for w in tokens if not w in stop_words]
frequency_dist2 = nltk.FreqDist(tokens)

# from wordcloud import WordCloud
# wordcloud = WordCloud().generate_from_frequencies(frequency_dist)
# plt.imshow(wordcloud)
# plt.axis("off")
# plt.show()

# wordcloud = WordCloud().generate_from_frequencies(frequency_dist2)
# plt.imshow(wordcloud)
# plt.axis("off")
# plt.show()

# print(tokens)
# text featuring ends
print(x)

# xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=0.2) # split the data for training and testing.

xTrain = []
xTest = []
yTrain = []
yTest = []
split = 0.95

for i in range(len(x)):

    if(i <= split*len(x)):
        xTrain.append(x[i])
        yTrain.append(y[i])
    else:
        xTest.append(x[i])
        yTest.append(y[i])    
    

# print(type(xTrain))
xTrain = np.array(xTrain)
xTest = np.array(xTest) #make an array of x test data
yTrain = np.array(yTrain) #make an array of y train data
yTest = np.array(yTest) #make an array of y test data

# print(xTrain)
# print(xTest)

vectorizer = TfidfVectorizer()
# x = v.fit_transform(df['Review'].values.astype('U'))
xTrain = vectorizer.fit_transform(xTrain)
xTest = vectorizer.transform(xTest)
test  = vectorizer.get_feature_names_out()
dfs = pd.DataFrame(test, columns=['Features'])
dfs.to_csv("importantFiles/test.csv", encoding='utf_8_sig')
print(test)

def logisticRegression(C,xTrain, yTrain, xTest, yTest): # train data by logistic Regression
    print("\nLogistic Regression, with C = {}__________________________________".format(C))
    model = LogisticRegression(C=C)
    model.fit(xTrain, yTrain)                       # train data
    print("slope = ", model.coef_)                       # get a slope here
    print("intercept = ", model.intercept_)              # get an intercept here
    print("train score = ", format(model.score(xTrain, yTrain)))
    print('\n')
    ypred = np.array(model.predict(xTest))   
    ypred = ypred.reshape(-1,1)            # make a tidy array of prediction data which contains values, -1, 0 or 1
    print(classification_report(yTest, ypred))
    print(confusion_matrix(yTest,ypred))


def linear_SVC (C, xTrain, yTrain, xTest, yTest):     # train data by SVC
    print("\nLinear SVC, with C = {}__________________________________".format(C))
    model = LinearSVC(C=C).fit(xTrain, yTrain)                         # train a data
    print("when C =", C)                                                    
    print("slope = ", model.coef_)                                          # get a slope here
    print("intercept = ", model.intercept_)                                 # get an intercept here
    print("mean accuracy = ", format(model.score(xTrain, yTrain)))
    print('\n')
    predData = np.array(model.predict(xTest))   
    predData = predData.reshape(-1,1)            # make a tidy array of prediction data which contains values, -1, 0 or 1
    ypred = np.array(model.predict(xTest))   
    ypred = ypred.reshape(-1,1)            # make a tidy array of prediction data which contains values, -1, 0 or 1
    print(classification_report(yTest, ypred))
    print(confusion_matrix(yTest,ypred))

def baseline_mostFrequent(xTrain, yTrain, xTest, yTest):
    print("\nbaseline mostfrequency classifier__________________________________")
    from sklearn.dummy import DummyClassifier
    dummy = DummyClassifier(strategy='most_frequent')
    dummy.fit(xTrain, yTrain)
    ypred = np.array(dummy.predict(xTest))   
    ypred = ypred.reshape(-1,1)            # make a tidy array of prediction data which contains values, -1, 0 or 1
    print(classification_report(yTest, ypred))
    print(confusion_matrix(yTest,ypred))

# mean_error=[]
# std_error = []
# cm=[]
# polyDegree = [1,2,4,5,6,8,10,40]
# for poly in polyDegree:
#     Xtrain = PolynomialFeatures(degree = poly).fit_transform(X)
#     ytrain = Y
#     model = LogisticRegression(penalty = "l2",C = 1, solver="lbfgs")

#     #5 fold CV
#     temp=[]
#     kf = KFold(n_splits=5)
#     for train, test in kf.split(Xtrain):
#         model.fit(Xtrain[train,:], ytrain[train])
#         ypred = model.predict(Xtrain[test,:])
#         temp.append(f1_score(ytrain[test],ypred))
#         Xnew = Xtrain[test,:]
#     mean_error.append(np.array(temp).mean())
#     std_error.append(np.array(temp).std())

logisticRegression(0.1,xTrain, yTrain, xTest, yTest)
# linear_SVC (0, xTrain, yTrain)
linear_SVC (0.01, xTrain, yTrain,  xTest, yTest)
linear_SVC (10, xTrain, yTrain,  xTest, yTest)
linear_SVC (100, xTrain, yTrain,  xTest, yTest)
baseline_mostFrequent(xTrain, yTrain,  xTest, yTest)
