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
# nltk.download('punkt')
# nltk.download('stopwords')
from sklearn.feature_extraction.text import TfidfVectorizer
from  sklearn.metrics  import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score
from sklearn.preprocessing import PolynomialFeatures

# #CAREFUL only for deprecation of datetime64
import warnings
warnings.filterwarnings("ignore")
# #CAREFUL

# df = pd.read_csv('importantFiles\Elon_class.csv')
df = pd.read_csv('/Users/doimasanari/Desktop/MachineLearning-GroupProject/importantFiles/Elon_class.csv')

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
dfs.to_csv("test.csv", encoding='utf_8_sig')
# dfs.to_csv("importantFiles/test.csv", encoding='utf_8_sig')
print(test)

def logisticRegression(C, xTrain, yTrain, xTest, yTest): # train data by logistic Regression
    print("\nLogistic Regression, with C = {}__________________________________".format(C))
    YTrain_buffer = []
    YTest_buffer = []

    #for class = 1
    for i in range(0, len(yTrain)):
        if(yTrain[i]==0):
            YTrain_buffer.append(-1)
        else:
            YTrain_buffer.append(yTrain[i])
    for i in range(0,len(yTest)):
        if(yTest[i]==0):
            YTest_buffer.append(-1)
        else:
            YTest_buffer.append(yTest[i])
    model1 = LogisticRegression(C=C,penalty = "l2")
    model1.fit(xTrain, YTrain_buffer)                       # train data

    #for class = 0
    for i in range(0, len(yTrain)):
        if(yTrain[i] == 1):
            YTrain_buffer[i] = -1
        elif(yTrain[i]==0):
            YTrain_buffer[i] = 1
    for i in range(0,len(yTest)):
        if(yTest[i] == 1):
            YTest_buffer[i] = -1
        elif(yTest[i]==0):
            YTest_buffer[i] = 1
    model2 = LogisticRegression(C=C,penalty = "l2")
    model2.fit(xTrain, YTrain_buffer)                       # train data

    #for class = -1
    for i in range(0, len(yTrain)):
        if(yTrain[i] == 1):
            YTrain_buffer[i] = -1
        elif(yTrain[i] == 0):
            YTrain_buffer[i] = -1
        elif(yTrain[i]==-1):
            YTrain_buffer[i] = 1
    for i in range(0,len(yTest)):
        if(yTest[i] == 1):
            YTest_buffer[i] = -1
        elif(yTest[i] == 0):
            YTest_buffer[i] = -1
        elif(yTest[i]==-1):
            YTest_buffer[i] = 1
    model3 = LogisticRegression(C=C,penalty = "l2")
    model3.fit(xTrain, YTrain_buffer)                       # train data

    ypred1 = np.array(model1.predict_proba(xTest)) #predicts 1->1 and 0,-1 -> -1
    ypred2 = np.array(model2.predict_proba(xTest)) #predicts 0->1 and 1,-1 -> -1
    ypred3 = np.array(model3.predict_proba(xTest)) #predicts -1->1 and 0,1 -> -1
    ypred = []
    for i in range(0, len(yTest)):
        if(ypred1[i][1]>ypred2[i][1] and ypred1[i][1]>ypred3[i][1]):
            ypred.append(1)
        elif(ypred2[i][1]>ypred1[i][1] and ypred2[i][1]>ypred3[i][1]):
            ypred.append(0)
        elif(ypred3[i][1]>ypred1[i][1] and ypred3[i][1]>ypred2[i][1]):
            ypred.append(-1)
        else:
            ypred.append(0)
    ypred = np.array(ypred)
    ypred = ypred.reshape(-1,1)            # make a tidy array of prediction data which contains values, -1, 0 or 1
    print(classification_report(yTest, ypred))
    print(confusion_matrix(yTest,ypred))

    mean_error=[]
    std_error = []
    C_values = [0.1, 1, 10, 100]
    for Ci in C_values:
        # xTrain = PolynomialFeatures(degree = Ci).fit_transform(x)
        model = LogisticRegression(penalty = "l2",C = Ci, solver="lbfgs")
        #5 fold CV
        temp=[]
        kf = KFold(n_splits=5)
        for train, test in kf.split(xTrain):        
            model.fit(xTrain[train], yTrain[train])
            ypred = model.predict(xTrain[test])
            temp.append(f1_score(yTrain[test],ypred,average = "micro"))
            Xnew = xTrain[test,:]
        mean_error.append(np.array(temp).mean())
        std_error.append(np.array(temp).std())
    plot(C_values, mean_error, std_error, True)
        # plt.errorbar(polyDegree, mean_error, yerr=std_error, ecolor ="red", marker = "o", ms=3)


def linear_SVC (C, xTrain, yTrain, xTest, yTest):     # train data by SVC
    print("\nLinear SVC, with C = {}__________________________________".format(C))
    YTrain_buffer = []
    YTest_buffer = []

    #for class = 1
    for i in range(0, len(yTrain)):
        if(yTrain[i]==0):
            YTrain_buffer.append(-1)
        else:
            YTrain_buffer.append(yTrain[i])
    for i in range(0,len(yTest)):
        if(yTest[i]==0):
            YTest_buffer.append(-1)
        else:
            YTest_buffer.append(yTest[i])
    model1 = LinearSVC(C=C, penalty= "l2")
    model1.fit(xTrain, YTrain_buffer)                       # train data

    #for class = 0
    for i in range(0, len(yTrain)):
        if(yTrain[i] == 1):
            YTrain_buffer[i] = -1
        elif(yTrain[i]==0):
            YTrain_buffer[i] = 1
    for i in range(0,len(yTest)):
        if(yTest[i] == 1):
            YTest_buffer[i] = -1
        elif(yTest[i]==0):
            YTest_buffer[i] = 1
    model2 = LinearSVC(C=C,penalty = "l2")
    model2.fit(xTrain, YTrain_buffer)                       # train data

    #for class = -1
    for i in range(0, len(yTrain)):
        if(yTrain[i] == 1):
            YTrain_buffer[i] = -1
        elif(yTrain[i] == 0):
            YTrain_buffer[i] = -1
        elif(yTrain[i]==-1):
            YTrain_buffer[i] = 1
    for i in range(0,len(yTest)):
        if(yTest[i] == 1):
            YTest_buffer[i] = -1
        elif(yTest[i] == 0):
            YTest_buffer[i] = -1
        elif(yTest[i]==-1):
            YTest_buffer[i] = 1
    model3 = LinearSVC(C=C,penalty = "l2")
    model3.fit(xTrain, YTrain_buffer)                       # train data

    ypred1 = np.array(model1._predict_proba_lr(xTest)) #predicts 1->1 and 0,-1 -> -1
    ypred2 = np.array(model2._predict_proba_lr(xTest)) #predicts 0->1 and 1,-1 -> -1
    ypred3 = np.array(model3._predict_proba_lr(xTest)) #predicts -1->1 and 0,1 -> -1
    ypred = []
    for i in range(0, len(yTest)):
        if(ypred1[i][1]>ypred2[i][1] and ypred1[i][1]>ypred3[i][1]):
            ypred.append(1)
        elif(ypred2[i][1]>ypred1[i][1] and ypred2[i][1]>ypred3[i][1]):
            ypred.append(0)
        elif(ypred3[i][1]>ypred1[i][1] and ypred3[i][1]>ypred2[i][1]):
            ypred.append(-1)
        else:
            ypred.append(0)
    ypred = np.array(ypred)
    ypred = ypred.reshape(-1,1)            # make a tidy array of prediction data which contains values, -1, 0 or 1
    print(classification_report(yTest, ypred))
    print(confusion_matrix(yTest,ypred))
    mean_error=[]
    std_error = []
    C_values = [0.1, 1, 10, 100]
    for Ci in C_values:
        model = LinearSVC(C=C, penalty= "l2").fit(xTrain, yTrain)
        #5 fold CV
        temp=[]
        kf = KFold(n_splits=5)
        for train, test in kf.split(xTrain):        
            model.fit(xTrain[train], yTrain[train])
            ypred = model.predict(xTrain[test])
            temp.append(f1_score(yTrain[test],ypred,average = "micro"))
            Xnew = xTrain[test,:]
        mean_error.append(np.array(temp).mean())
        std_error.append(np.array(temp).std())
    plot(C_values, mean_error, std_error, False)

def baseline_mostFrequent(xTrain, yTrain, xTest, yTest):
    print("\nbaseline mostfrequency classifier__________________________________")
    from sklearn.dummy import DummyClassifier
    dummy = DummyClassifier(strategy='most_frequent')
    dummy.fit(xTrain, yTrain)
    ypred = np.array(dummy.predict(xTest))   
    ypred = ypred.reshape(-1,1)            # make a tidy array of prediction data which contains values, -1, 0 or 1
    print(classification_report(yTest, ypred))
    print(confusion_matrix(yTest,ypred))


def plot(c, mean_error, std_error, isLogistic):
    plt.errorbar(c, mean_error, yerr=std_error, ecolor ="red", marker = "o", ms=3)
    # plt.xlabel("Degree of polynomial")
    plt.ylabel("f1 score")
    if(isLogistic == True):
        plt.title("Cross Validation in Logistic regression, C = " + str(c))
    else:
        plt.title("Cross Validation in LinearSVC, C = " + str(c))
    plt.show()


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


# #CHOSEN ORDER OF POLYNOMIAL
# deg = 2
# #plot data
# plt.errorbar(polyDegree, mean_error, yerr=std_error, ecolor ="red", marker = "o", ms=3)
# plt.xlabel("Degree of polynomial")
# plt.ylabel("f1 score")
# plt.title("Logistic regression")
# plt.show()


logisticRegression(0.1, xTrain, yTrain, xTest, yTest)
# linear_SVC (0, xTrain, yTrain)
linear_SVC (0.001, xTrain, yTrain,  xTest, yTest)
linear_SVC (0.1, xTrain, yTrain,  xTest, yTest)
linear_SVC (1, xTrain, yTrain,  xTest, yTest)
linear_SVC (70, xTrain, yTrain,  xTest, yTest)
linear_SVC (90, xTrain, yTrain,  xTest, yTest)
baseline_mostFrequent(xTrain, yTrain,  xTest, yTest)
