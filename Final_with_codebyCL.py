
# -*- coding: utf-8 -*-

# Sample coding of a final project that involves some basic natural language processing, linear regression, 
# and random forests. I partnered with another classmate for this project.

# Remember to cite WWW / SIGIR papers if you use the data:
# R. He, J. McAuley. Modeling the visual evolution of fashion trends with one-class collaborative filtering. WWW, 2016
# J. McAuley, C. Targett, J. Shi, A. van den Hengel. Image-based recommendations on styles and substitutes. SIGIR, 2015

# Created on Sun Jun  4 13:02:01 2017
# @author: Sheah, Christina 

#--------------------------------------

import pandas as pd
import gzip

import os
os.getcwd()
os.chdir(r'C:\Users\ChristinaL\Documents\Econ 294A Python Lab')
os.getcwd()
#path = "C:\\Users\\Sheah\\Documents\\Python Lab"

def parse(path):
  g = gzip.open(path, 'rb')
  for l in g:
    yield eval(l)

def getDF(path):
  i = 0
  df = {}
  for d in parse(path):
    df[i] = d
    i += 1
  return pd.DataFrame.from_dict(df, orient='index')

#df = getDF('reviews_Beauty.json.gz')
#print(df.head())

df1 = getDF('meta_Beauty.json.gz')
print(df1.head())

df2 = getDF('reviews_Beauty_5.json.gz')
result = pd.merge(df1.reset_index(), df2.reset_index(), on=['asin'], how='inner').set_index(['asin','reviewerID'])
print(result.head())

#result.to_csv(result, sep =' ', index = False, header = False, encoding = 'utf-8')

result['reviewText'] = result['reviewText'].fillna('')
result['reviewcount'] = result['reviewText'].str.split()
print(result['reviewcount'].head(20))
type(result['reviewcount'])

result['reviewlen'] = result.apply(lambda row: len(row['reviewcount']), axis=1)

print(result['reviewlen'].head())

print(result.corr()['overall'])

print(result['index_x'].head)

print(result['index_y'].head)

from sklearn.decomposition import PCA


import nltk
from nltk.book import *
type(text1)
text1
def lexical_diversity(text): 
    if len(set(text)) > 0:
         return len(text) / len(set(text)) 
     

result['lexicaldiversity'] = result.apply(lambda row: lexical_diversity(row['reviewcount']), axis = 1)

import collections, re


#result['poscount'] = result.apply lambda row: 
#searchwords = set(['chuck', 'peter', 'piper', 'peck', 'pick', 'picked'])
#cnt = collections.Counter()
#words = re.findall('\w+', text.lower())

#def wordcount(text):
#    for word in words:
#        if word in searchwords:
#        cnt [word] += 1
#            return cnt

print(result['lexicaldiversity'].head())
print(result['reviewcount'].head())
import pandas as pd
from sklearn import datasets, linear_model
columns = result.columns.tolist()
columns = [c for c in columns if c in ["unixReviewTime", "reviewlen", "price", "lexicaldiversity"]]
type(result["unixReviewTime"])
type(result["reviewlen"])
type(result["overall"])
target = "overall"

print(result.head())
import seaborn as sns
corr = result[columns].corr()
sns.heatmap(corr, xticklabels = corr.columns.values, yticklabels = corr.columns.values)

import matplotlib.pyplot as plt
plt.scatter(result['price'], result['overall'])
plt.show()
plt.scatter(result['reviewlen'], result['overall'])
plt.show()
plt.scatter(result['price'], result['reviewlen'])
plt.show()

plt.hist(result['reviewlen'], bins=100)
plt.show()

       
train = result.sample(frac=.8, random_state=1)
test = result.loc[~result.index.isin(train.index)]


print(train.shape)
print(test.shape)
train = train.dropna()
test = test.dropna()

print(train.shape)
print(test.shape)

#Let's fit with linear regression now
model = linear_model.LinearRegression()
model.fit(train[columns], train[target])

from sklearn.metrics import mean_squared_error
predictions = model.predict(test[columns])
print(predictions)
plt.hist(predictions)

plt.hist(result['overall'].dropna())

#Random Forest!
mean_squared_error(predictions, test[target])

from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(n_estimators=100, min_samples_leaf=10, random_state=1)

model.fit(train[columns], train[target])

predictions = model.predict(test[columns])

mean_squared_error(predictions, test[target])



# ============================ Code from Christina ==========================
# could not control for robustness due to memory error

import statsmodels.formula.api as sm

# Some basic regressions --> linear regression
regOverall1 = sm.ols(formula="overall ~ price",data=result).fit()
print(regOverall1.params)
print(regOverall1.summary())


regOverall2 = sm.ols(formula="overall ~ reviewlen",data=result).fit()
print(regOverall2.params)
print(regOverall2.summary())
# negative relationship as we expected


regOverall3 = sm.ols(formula="overall ~ lexicaldiversity",data=result).fit()
print(regOverall3.params)
print(regOverall3.summary())
# negative relationship as we expected


# count of positive words in each review
import collections, re
def pos_wordcount(text):
    poswords = set(['good', 'liked', 'pretty', 'beautiful', 'cute', 'fantastic', 'great', 'happy', 'easy', 'amazing', 'incredible', 'satisfied'])
    if text != '':
        words = re.findall('\w+', text.lower())
        cnt =0
        for word in words:
            if word in poswords:
                cnt = cnt + 1
        return cnt
   
result['posWordCount'] = result.apply(lambda row: pos_wordcount(row['reviewText']), axis=1)

# linear regression of overall on positive words count
reg_Overallpos = sm.ols(formula="overall ~ posWordCount",data=result).fit()
print(reg_Overallpos.params)
print(reg_Overallpos.summary())


# count of negative words in each review
import collections, re
def neg_wordcount(text):
    negwords = set(['terrible', 'horrible', 'bad', 'sucked', 'chalky', 'hard', 'uncomfortable', 'ugly', 'unsatisfied', 'unhappy'])
    if text != '':
        words = re.findall('\w+', text.lower())
        cnt =0
        for word in words:
            if word in negwords:
                cnt = cnt + 1
        return cnt
    
result['negWordCount'] = result.apply(lambda row: neg_wordcount(row['reviewText']), axis=1)
#print(result['negWordCount'].head(100))
#print(max(result['negWordCount']))

# linear regression of overall on negative words count
reg_Overallneg = sm.ols(formula="overall ~ negWordCount",data=result).fit()
print(reg_Overallneg.params)
print(reg_Overallneg.summary())

# some plots to observe the relationships
plt.plot(result['posWordCount'],result['overall'],'ro')
plt.suptitle("Figure 1: Plot of Overall on Positive Word Count")
#plt.xticks(rotation=45)
plt.xlabel('Positive Words Count')
plt.ylabel('Overall')
plt.show()

plt.plot(result['negWordCount'],result['overall'],'ro')
plt.suptitle("Figure 2: Plot of Overall on Negative Word Count")
#plt.xticks(rotation=45)
plt.xlabel('Negative Words Count')
plt.ylabel('Overall')
plt.show()
