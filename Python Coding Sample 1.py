# -*- coding: utf-8 -*-
"""
Created on Tue May 16 23:36:09 2017

Econ 294A Python Lab Problem Set 5
@author: Christina Louie
May 21, 2017
"""
# ---- library -------
import collections, re

# ------------------ PART I ------------------------
# Create a list of length five for each of the following types of tokens:
    
# (a)
listofunigrams = ['snow','charming','happy','sleepy','dopey']
print(len(listofunigrams))
# 5

# (b)
listoftrigrams = ['young and hungry','switched at birth','the green arrow','the boss baby','the american housewife']
print(len(listoftrigrams))
# 5


# ------------------ PART II ------------------------
# Using the Collections library in Python, count the frequency of the 
# listed unigrams in each of the following tongue twister text:
    
# (a)
text = 'Through three cheese trees three free fleas flew. While these fleas flew, freezy breeze blew. Freezy breeze made these three trees freeze. Freezy trees made these trees\' cheese freeze. That\'s what made these three free fleas sneeze.'

wordstosearch = set(['three','trees','freezy'])
cnt = collections.Counter()
words = re.findall('\w+',text.lower())
print(words)
for word in words:
    if word in wordstosearch:
        cnt[word] = cnt[word] + 1
print(cnt)
# output: Counter({'three': 4, 'trees': 4, 'freezy': 3})

#(b)

secondtext = 'Betty Botter bought some butter But she said the butter\'s bitter If I put it in my batter, it will make my batter bitter But a bit of better butter will make my batter better So \'twas better Betty Botter bought a bit of better butter'

searchwords = set(['butter','botter','better'])
count = collections.Counter()
wordtext = re.findall('\w+',secondtext.lower())
print(wordtext)
for word in wordtext:
    if word in searchwords:
        count[word] = count[word] + 1
print(count)
# output: Counter({'butter': 4, 'better': 4, 'botter': 2})


# ------------------ PART III ------------------------
'''
Using a dataset, train predictive models then test the accuracy of your 
predictions. Make sure to train at least two predictive models. 
'''
import os 
os.chdir(r'C:\Users\ChristinaL\Documents\Econ 294A Python Lab')

# import panda
import pandas as pd

# upload dataset
abalonedata = pd.read_csv(r'abalone.csv',low_memory = False)

# To check the variable names, we can check the column names
print(abalonedata.columns)

# We can also see the shape of the data (number of rows and columns):
print(abalonedata.shape)

# ------- linear regression model --------------
# We first want to assess which variables may be useful in ‘predicting’
# rings. To do this, we can
# check the correlation of the other variables to average_rating:
print(abalonedata.corr()["rings"])

# we see that shell_weight and diameters have the largest correlations

# We then remove columns we think are not very useful (especially those with texts or the dependent 
# variable):
# Get all the columns from the dataframe.
columns = abalonedata.columns.tolist()

# Filter the columns to remove ones we don't want
columns = [c for c in columns if c not in ["rings","sex"]]

# Store the variable we'll be predicting on
target = "rings"

# randomly split our data with 80% as the train set and 20% as the test set:
train = abalonedata.sample(frac=0.8, random_state=1) 
# select anything not in the training set and put it in the testing set
test = abalonedata.loc[~abalonedata.index.isin(train.index)]
#print shape
print(train.shape)
print(test.shape)


# USE LINEAR REGRESSION MODEL TO FIT DATA

# Import the linearregression model
from sklearn.linear_model import LinearRegression
# Initialize the model class
model = LinearRegression()
# Fit the model to the training data. 
model.fit(train[columns],train[target])


# We then predict the error. In doing so, the test data must have the same format as the train data or 
# we may lose accuracy.
from sklearn.metrics import mean_squared_error
# Generate prediction for test set.
predictions = model.predict(test[columns])
# compute error between test predictions and the actual values
print(mean_squared_error(predictions,test[target]))

# output: 4.4933293377


# -------- Random forest regressor Model ----------
from sklearn.ensemble import RandomForestRegressor
# Initialize the model with some parameters.
model = RandomForestRegressor(n_estimators=100, min_samples_leaf=10,random_state=1)
# Fit the model to the data.
model.fit(train[columns], train[target])
# Make prediction
predictions = model.predict(test[columns])
# Compute the error
print(mean_squared_error(predictions,test[target]))

# output: 4.1440641387











