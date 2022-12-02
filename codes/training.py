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
from yellowbrick.classifier import ROCAUC



# #CAREFUL only for deprecation of datetime64
import warnings
warnings.filterwarnings("ignore")
# #CAREFUL

df = pd.read_csv('importantFiles\Elon_class.csv')
# df = pd.read_csv('/Users/doimasanari/Desktop/MachineLearning-GroupProject/importantFiles/Elon_class.csv')

x = df['Tweet']  # construct a matrix containing tweets
y = df['Class']  # construct a matrix containing -1, 0 or 1
date = df['Date'] # construct a matrix containing the date

# Tweets = df['Tweet'].str.cat(sep=' ')
# #function to split text into word
# tokens = word_tokenize(Tweets)
# vocabulary = set(tokens)
# print(len(vocabulary))
# frequency_dist = nltk.FreqDist(tokens)
# print(sorted(frequency_dist,key=frequency_dist.__getitem__, reverse=True)[0:50])
# stop_words = set(stopwords.words('english'))
# tokens = [w for w in tokens if not w in stop_words]
# frequency_dist2 = nltk.FreqDist(tokens)

# from wordcloud import WordCloud
# wordcloud = WordCloud().generate_from_frequencies(frequency_dist2)
# plt.imshow(wordcloud)
# plt.axis("off")
# plt.show()

print(x)

# xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=0.2) # split the data for training and testing.

xTrain = []
xTest = []
yTrain = []
yTest = []
split = 0.85

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

vectorizer = TfidfVectorizer( lowercase=False, ngram_range=(2,2))
# x = v.fit_transform(df['Review'].values.astype('U'))
xTrain = vectorizer.fit_transform(xTrain)
xTest = vectorizer.transform(xTest)
test  = vectorizer.get_feature_names_out()
dfs = pd.DataFrame(test, columns=['Features'])
# dfs.to_csv("test.csv", encoding='utf_8_sig')
# dfs.to_csv("importantFiles/test.csv", encoding='utf_8_sig')
print(test[0:50])
coef=vectorizer.idf_
coef_feat = vectorizer.get_feature_names()
coef,coef_feat = (list(t) for t in zip(*sorted(zip(coef, coef_feat))))
size = len(coef)
# print(coef[0:50])
# print(coef[size-50:size])
# print (dict(zip(vectorizer.get_feature_names(), vectorizer.idf_)))

def plot_ROC_curve(model, xtrain, ytrain, xtest, ytest):

    # Creating visualization with the readable labels
    visualizer = ROCAUC(model, encoder={2: 'Stock increase', 
                                        0: 'nothing happens', 
                                        -2: 'Stock decreaess'})
                                        
    # Fitting to the training data first then scoring with the test data                                    
    visualizer.fit(xtrain, ytrain)
    visualizer.score(xtest, ytest)
    visualizer.show()
    
    return visualizer

def logisticRegression_CV(C, xTrain, yTrain, xTest, yTest): # train data by logistic Regression
    print("\nLogistic Regression, with C = {}__________________________________".format(C))
    
    #Cross validation of LR
    mean_error=[]
    std_error = []
    count = len(C)
    for Ci in C:
        print("progress: {}".format(count))
        count=count-1
        # xTrain = PolynomialFeatures(degree = Ci).fit_transform(x)
        model1 = LogisticRegression(penalty = "l2",C = Ci,multi_class='ovr',class_weight='balanced')
        #5 fold CV
        temp=[] 
        kf = KFold(n_splits=5)
        ytrain1 = np.array(yTrain)
        for train, test in kf.split(yTrain): 
            
            # train one vs all models
            model1.fit(xTrain[train],ytrain1[train])
            
            #predict one vs all modelss
            ypred1 = np.array(model1.predict(xTrain[test])) #predicts 1->1 and 0,-1 -> -1
    
            temp.append(f1_score(yTrain[test],ypred1,average = "micro"))
            
        mean_error.append(np.array(temp).mean())
        std_error.append(np.array(temp).std())
    
    model1 = LogisticRegression(penalty = "l2",C = 10**5,multi_class='ovr',class_weight='balanced')
    model1.fit(xTrain,yTrain)
    ypred = model1.predict(xTest)


    print(classification_report(yTest, ypred))
    print(confusion_matrix(yTest,ypred))
    plot_ROC_curve(model1,xTrain, yTrain, xTest, yTest)
    plot(C, mean_error, std_error, True)


def linear_SVC_CV (C, xTrain, yTrain, xTest, yTest):     # train data by SVC
    print("\nLinear SVC, with C = {}__________________________________".format(C))
    
    #Cross validation of LR
    mean_error=[]
    std_error = []
    count = len(C)
    for Ci in C:
        print("progress: {}".format(count))
        count=count-1
        # xTrain = PolynomialFeatures(degree = Ci).fit_transform(x)
        model1 = LinearSVC(penalty = "l2",C = Ci, multi_class='ovr',class_weight='balanced')
     
        #5 fold CV
        temp=[] 
        
        kf = KFold(n_splits=5)

        for train, test in kf.split(yTrain): 
            ytrain1 = np.array(yTrain)
            
            # train one vs all models
            model1.fit(xTrain[train],ytrain1[train])
            
            #predict one vs all modelss
            ypred1 = np.array(model1.predict(xTrain[test])) #predicts 1->1 and 0,-1 -> -1
    
            temp.append(f1_score(yTrain[test],ypred1,average = "micro"))
        mean_error.append(np.array(temp).mean())
        std_error.append(np.array(temp).std())

    model1 = LinearSVC(penalty = "l2",C = 10**5, multi_class='ovr',class_weight='balanced')
    model1.fit(xTrain,yTrain)
    ypred = np.array(model1.predict(xTest))
    print(classification_report(yTest, ypred))
    print(confusion_matrix(yTest,ypred))
    plot_ROC_curve(model1,xTrain, yTrain, xTest, yTest)
    plot(C, mean_error, std_error, False)

def baseline_mostFrequent(xTrain, yTrain, xTest, yTest):
    print("\nbaseline mostfrequency classifier__________________________________")
    from sklearn.dummy import DummyClassifier
    dummy = DummyClassifier(strategy='most_frequent')
    dummy.fit(xTrain, yTrain)
    ypred = np.array(dummy.predict(xTest))   
    ypred = ypred.reshape(-1,1)            # make a tidy array of prediction data which contains values, -1, 0 or 1
    print(classification_report(yTest, ypred))
    print(confusion_matrix(yTest,ypred))
    plot_ROC_curve(dummy,xTrain, yTrain, xTest, yTest)


def plot(c, mean_error, std_error, isLogistic):
    print("ploting...")
    plt.errorbar(c, mean_error, yerr=std_error, ecolor ="red", marker = "o", ms=3)
    # plt.xlabel("Degree of polynomial")
    plt.ylabel("f1 score")
    if(isLogistic == True):
        plt.title("Cross Validation in Logistic regression, C = " + str(c))
    else:
        plt.title("Cross Validation in LinearSVC, C = " + str(c))
    plt.show()

# def plot_top_features(classifier):
#     model_coefs = pd.DataFrame(classifier.coef_)
#     coefs_df = model_coefs.T
#     feature_name_list =  
#     # list w/ eng'd features & tf-idf n-grams
#     all_feat_names = []
#     for i in feature_name_list:
#         all_feat_names.append(i)
#     for i in tfidf.get_feature_names():
#         all_feat_names.append(i)
#     # creating column for feat names
#     coefs_df['feats'] = all_feat_names
#     coefs_df.set_index('feats', inplace=True)
#     coefs_df['feats'] = all_feat_names
#     coefs_df.set_index('feats', inplace=True)
#     # plot non-cb
#     coefs_df[0].sort_values(ascending=True).head(20).plot(kind='bar')
#     plt.title("SVM: Top 20 Non-Clickbait Coefs")
#     # plt.title("LogReg: Top 20 Non-Clickbait Coefs")
#     plt.xlabel("features")
#     plt.ylabel("coef value")
#     plt.xticks(rotation=55)
#     plt.show()
#     # plot CB classification
#     coefs_df[0].sort_values(ascending=False).head(20).plot(kind='bar', color='orange')
#     plt.title("SVM: Top 20 Clickbait Coefs")
#     # plt.title("LogReg: Top 20 Clickbait Coefs")
#     plt.xlabel("features")
#     plt.ylabel("coef value")
    # plt.xticks(rotation=55)
    # plt.show()


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




baseline_mostFrequent(xTrain, yTrain,  xTest, yTest)

logisticRegression_CV([10**-2,10**-1,10**0,10,50,10**5], xTrain, yTrain, xTest, yTest)
# linear_SVC (0, xTrain, yTrain)
linear_SVC_CV ([10**-2,10**-1,10**0,10,50,10**5], xTrain, yTrain,  xTest, yTest)

