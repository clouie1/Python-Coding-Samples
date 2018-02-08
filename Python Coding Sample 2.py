# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 22:12:50 2017

Econ 294A Python Lab 
@author: Christina Louie
April 30, 2017
"""
# ---- library -------
import sys

# ------------------ PART I ------------------------

# (a)
"""
The average of the remainder of n different ratios (You may assume that 
the ratios are created from two lists of length n, where the ratios are 
created with the numbers corresponding to the same index).

Our ending goal is to have the average of the remainder of n different 
ratios (assume that the ratios are created from two different lists of length n).
Step 1: check if the the two lists match in length. 
Step 2: We need to find the remainder of the ratios. Then put the 
remainders into a list. 
Step 3: We calculate the total of the remainder 
then divide it by the length of the list of remainders. 
"""
    
def avgOfRemainders(list1,list2):
    # check if the two lists match in length
    if len(list1) == len(list2):
        pass
    else:
        print("The length do not match.")
        sys.exit()
    # create a list of the remainder of the ratio
    listOfRatioRemain = []
    x=0
    while x < len(list1):
        num = (list1[x])%(list2[x])
        listOfRatioRemain.append(num)
        x = x + 1
        
    total=0
    # find the average of the ratios of remainder
    for nums in listOfRatioRemain:
        total = total + nums
    avg = total/len(listOfRatioRemain)
    print(avg)
        
#an example     
list1 = [21,22,23,27,28]
list2 = [5,3,3,7,3]
avgOfRemainders(list1,list2)

# output: 2.2

# (b)
"""
Using the Pandas dataframe, determine 20 days which the "adjusted closing 
price" of the Netflix stock dropped the most from May 24, 2002 to the end
of 2016.

Step 1: Install a package into python. (Installed Already)
Step 2: Download the netflix data
Step 3: Build a Pandas Dataframe with the data
Step 4: Create the lagged value
Step 5: Sort data by the change in closing price
Step 6: Obtain the top 20 rows of the change 
"""
# Import a module ('Share') from the yahoo_finance package
from yahoo_finance import Share

# call the netflix share
Netflix = Share('NFLX')

# acquire historical data from May 24, 2002 to the 
# end of 2016 (December 31, 2016).
NetflixHistorical = Netflix.get_historical('2002-05-24', '2016-12-31')

# This provides lists we can use to combine separate data together
datelist =[]
adjclosepricelist =[]

# This loop saves each data into their corresponding list
for listcomp in NetflixHistorical:
    date = listcomp['Date']
    adjustcloseprice = listcomp['Adj_Close']
    datelist.append(date)
    adjclosepricelist.append(adjustcloseprice)

# This combines the lists into a single list:
Netflixinfo = list(zip(datelist, adjclosepricelist))

# create Pandas dataframe
import pandas as pd
dfNetflixHistorical = pd.DataFrame(data=Netflixinfo, columns=['Date','Adjusted Closing Price'])

### Create the lagged value
# The following creates a Pandas dataframe with the lag values:
# This creates a Pandas dataframe from the column "Adjusted Closing Price":
df_adjclosepricelist = dfNetflixHistorical[['Adjusted Closing Price']]

# This creates another dataframe that has values equivalent to the lag of the specified dataframe
lagvalue_adjclosepricelist = df_adjclosepricelist.shift()

# Rename the column:
lagvalue_adjclosepricelist = lagvalue_adjclosepricelist.rename(columns={'Adjusted Closing Price': 'Lag Adjusted Closing Price'})

# Combine the two pandas dataframe:
df_adjcloseprandlag = pd.concat([dfNetflixHistorical, lagvalue_adjclosepricelist], axis=1)

# This drops all the rows with NaN values
df_adjcloseprandlag = df_adjcloseprandlag.dropna()

# This creates the change in the adjusted closing price
df_adjcloseprandlag['changeinadjclosepr'] = df_adjcloseprandlag['Adjusted Closing Price'].astype(float) - df_adjcloseprandlag['Lag Adjusted Closing Price'].astype(float)

# Sort data by the change in closing price 
sort_df = df_adjcloseprandlag.sort_values(['changeinadjclosepr'], ascending=True)

# To obtain the first 20 rows:
print(sort_df.head(n=20))

# Output:
#           Date Adjusted Closing Price Lag Adjusted Closing Price  \
#52   2016-10-17              99.800003                 118.790001   
#370  2015-07-15              98.129997                 115.809998   
#433  2015-04-15              67.922859                  80.292854   
#250  2016-01-05             107.660004                     117.68   
#492  2015-01-20              49.828571                  58.468571   
#341  2015-08-25             101.519997                 110.129997   
#357  2015-08-03             112.559998                 121.150002   
#742  2014-01-22              47.675713                  55.531429   
#284  2015-11-13             103.650002                 111.349998   
#340  2015-08-26             110.129997                 117.660004   
#311  2015-10-07             108.099998                     114.93   
#993  2013-01-23              14.751429                      20.98   
#932  2013-04-22                  24.91                  30.998571   
#283  2015-11-16             111.349998                 117.099998   
#222  2016-02-16              89.050003                  94.760002   
#321  2015-09-23                  98.07                 103.760002   
#18   2016-12-05             119.160004                     124.57   
#127  2016-06-30              91.480003                  96.669998   
#314  2015-10-02             106.110001                     111.25   
#327  2015-09-15              99.160004                 104.080002   
#
#     changeinadjclosepr  
#52           -18.989998  
#370          -17.680001  
#433          -12.369995  
#250          -10.019996  
#492           -8.640000  
#341           -8.610000  
#357           -8.590004  
#742           -7.855716  
#284           -7.699996  
#340           -7.530007  
#311           -6.830002  
#993           -6.228571  
#932           -6.088571  
#283           -5.750000  
#222           -5.709999  
#321           -5.690002  
#18            -5.409996  
#127           -5.189995  
#314           -5.139999  
#327           -4.919998  

# (c)
"""
Using the Pandas dataframe, search for the median, maximum, and minimum values
of the opening price of the Tesla stock in 2016.

Step 1: Download the Tesla data
Step 2: Build a Pandas Dataframe with the data
Step 3: Sort data in ascending order
Step 4: Search for the median, maximumm and minimum 
"""

# Import a module ('Share') from the yahoo_finance package
from yahoo_finance import Share

# call the netflix share
Tesla = Share('TSLA')

# acquire historical data 
TeslaHistorical = Tesla.get_historical('2016-01-01', '2016-12-31')

# This provides lists we can use to combine separate data together
datelist =[]
openpricelist =[]

# This loop saves each data into their corresponding list
for listcomp in TeslaHistorical:
    date = listcomp['Date']
    openprice = listcomp['Open']
    #print(adjustcloseprice)
    datelist.append(date)
    openpricelist.append(openprice)

# Combine the lists into a single list:
Teslainfo = list(zip(datelist, openpricelist))

# Create Pandas dataframe
import pandas as pd
dfTeslaHistorical = pd.DataFrame(data=Teslainfo, columns=['Date','Open Price'])

# This drops all the rows with NaN values
dfTeslaHistorical = dfTeslaHistorical.dropna()

# Sort data in ascending order by open price 
sort_df = dfTeslaHistorical.sort_values(['Open Price'], ascending=True)

# Search for median, maximum, and minimum
median = pd.DataFrame.median(sort_df['Open Price'])
print(median)
maximum = pd.DataFrame.max(sort_df['Open Price'])
print(maximum)
minimum = pd.DataFrame.min(sort_df['Open Price'])
print(minimum)

# Output:
# Median: 209.0999985
# Max: 266.450012
# Min: 142.320007

# ------------------ PART II ------------------------
# (a)
from yahoo_finance import Share

# call the Google share
Google = Share('GOOG')

# acquire historical data from January 1, 2011 to the 
# end of 2016 (December 31, 2016).
GoogleHistorical = Google.get_historical('2011-01-01', '2016-12-31')

# This provides lists we can use to combine separate data together
datelist =[]
adjclosepricelist =[]

# This loop saves each data into their corresponding list
for listcomp in GoogleHistorical:
    date = listcomp['Date']
    adjustcloseprice = listcomp['Adj_Close']
    datelist.append(date)
    adjclosepricelist.append(adjustcloseprice)

# This combines the lists into a single list:
Googleinfo = list(zip(datelist, adjclosepricelist))

# (b)
# create Pandas dataframe
import pandas as pd
dfGoogleHistorical = pd.DataFrame(data=Googleinfo, columns=['Date','Adjusted Closing Price'])

# Create the lagged value
df_adjclosepricelist = dfGoogleHistorical[['Adjusted Closing Price']]

# This creates another dataframe that has values equivalent to the lag of the specified dataframe
lagvalue_adjclosepricelist = df_adjclosepricelist.shift()

# Rename the column:
lagvalue_adjclosepricelist = lagvalue_adjclosepricelist.rename(columns={'Adjusted Closing Price': 'Lag Adjusted Closing Price'})

# Combine the two pandas dataframe:
df_GoogleWithLag = pd.concat([dfGoogleHistorical, lagvalue_adjclosepricelist], axis=1)

# This drops all the rows with NaN values
df_GoogleWithLag = df_GoogleWithLag.dropna()

# (c)
# This creates the change in the adjusted closing price
df_GoogleWithLag['changeinadjclosepr'] = df_GoogleWithLag['Adjusted Closing Price'].astype(float) - df_GoogleWithLag['Lag Adjusted Closing Price'].astype(float)

# (d) plot the time series of the difference in the adjusted closing price
df_GoogleWithLag.plot(x='Date', y='changeinadjclosepr', style='-')


# -------------------- EXTRA CREDIT --------------------------
from datetime import datetime
import random
import pandas as pd

def extracredit(N):
    x_datelist=[]
    y_list =[]
    y=0
    x=0
    while x < N:
        year = random.randint(2000,2017)
        month = random.randint(1,12)
        day = random.randint(1,28)
        random_date = datetime(year, month, day)
        x_datelist.append(random_date)
        x=x + 1
    while y < N:
        random_value = random.randint(0,100)
        y_list.append(random_value)
        y=y + 1
    
    xyinfo = list(zip(x_datelist, y_list))
    df_xyinfo = pd.DataFrame(data=xyinfo, columns=['Date','Y Values'])
    print(df_xyinfo)
    variance = pd.DataFrame.var(df_xyinfo['Y Values'])
    print(variance)

extracredit(20)
# The output is the variance. It will be different everytime the code is ran.      


