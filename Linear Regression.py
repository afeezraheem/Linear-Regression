# -*- coding: utf-8 -*-
"""
Created on Tue Sep  8 21:37:40 2020

@author: jider
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df= pd.read_csv('vehicles.csv')

df.columns

df.describe()

#Eliminate the duplicates

df.drop_duplicates(inplace= True)

#Remove the nulls

df.isnull().any()

df.isnull().sum()/df.shape[0]

#Alternatively, remove null columns over a particular threshold- i.e. 60% of the total rows

thresh = len(df)*.6

df.dropna(thresh= thresh, axis= 1)

df.dropna(thresh= thresh, axis= 1).shape
df.dropna(thresh= 21, axis = 0).shape

# To convert everything here to all lower or upper

df.description.head()
df.description.astype(str).apply(lambda x: x.lower())

df.dtypes

#To calculate the length of the text | Descriptions > 1 or <=1 can be a dummy variable

df['text_len']= df.description.apply(lambda x: len(str(x)))

(df['text_len'].value_counts() > 1).sum()

---# To know the data type..It is a mixed type/object

     df.cylinders.dtype
     
---#To change to numerical types

     df.cylinders.value_counts()
     
df.cylinders= df.cylinders.apply(lambda x: str(x).lower().replace('cylinders','').strip())
     
     df.cylinders.value_counts()
     
---#To convert "others" category to Nan. That is, invalid parsing sets to NaN

    df.cylinders= pd.to_numeric(df.cylinders, errors= 'coerce')
    df.cylinders.value_counts()
    
# For the dependent variable, price, is it highly skewed?

 df.boxplot('price')
 
 df.boxplot('odometer')
 
 df.price.max()
 
 df.odometer.max()
 
# To trim the outliers using the df_outliers dataframe
 
 df_outliers= df[(df.price< df.price.quantile(.995)) & (df.price > df.price.quantile(.005))]
 
 df_outliers= df_outliers[(df_outliers.odometer< df_outliers.odometer.quantile(.995)) & (df_outliers.odometer > df_outliers.odometer.quantile(.005))]
 
#To see the histograms ~ Only 'Cylinders' is normally distributed. OLS Assumption- Residuals of outcome Y should

df_outliers[['price', 'odometer','cylinders','text_len']].hist()

# For the null values 

df_outliers.isnull().sum()/df_outliers.shape[0]

# To drop the zero subsets of variables with less nulls, such as manufacturer, year, make,fuel, transmission, title status

df_outliers.dropna(subset=['manufacturer','year','model','fuel','title_status','transmission'],inplace= True)
 
df_outliers.isnull().sum()/df_outliers.shape[0]

#As Cylinder does not contain significant % of null(per earlier thresh),I'd fill the nulls with the median value

df_outliers.cylinders.fillna(df_outliers.cylinders.median(), inplace= True)

df_outliers.isnull().sum()/df_outliers.shape[0]

#For the categorical variables, fill the nulls with 'n/a'

df_outliers[['vin','condition','drive','type','paint_color']]= df_outliers[['vin','condition','drive','type','paint_color']].fillna('n/a')

df_outliers.isnull().sum()/df_outliers.shape[0]

#Changing the vin categorical variable to 'has vin' and 'no vin' as the numbers are not helpful

df_outliers.vin= df_outliers.vin.apply(lambda x: 'has vin' if 'na' else 'no vin')

# I drop the variables that are less important in determining the price

df_final_vars= df_outliers.drop(['description','id','county','size','region','region_url','url','lat', 'long','image_url'], axis= 1)

-------//Regression\\-------
--------------------\\Using the sklearn linear regression & Stat model OLS\\---

df_final_vars['constant']= 1

# Introducing vehicle age instead of year alone. It helps to have the real age

df_final_vars['age']= 2019-df_final_vars.year

df_final_vars.isnull().any()

# To check for possible correlations between the independent variables

numeric = df_final_vars._get_numeric_data()

import seaborn as sns

corrdf= numeric

corr = corrdf.corr()

ax= sns.heatmap(
    
    corr,
    
    vmin=-1, vmax=1, center =0,
    
    cmap = sns.diverging_palette(20, 220, n=200),
    square = True
)

ax.set_xticklabels(
    
    ax.get_xticklabels(),
    rotation = 45,
    horizontalalignment = 'right'
);

# Completing the Linear Regression for each variable- odometer to start with. I include math for RMSE

from sklearn.model_selection import train_test_split
from sklearn.linear_model import  LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math


# Adjust variables to a specific format >numpy array>> array of arrays. Otherwise, it won't run in the sklearn library

y1= df_final_vars.price.values.reshape(-1,1)

x1= df_final_vars.odometer.values.reshape(-1,1)

# Split into a train and test set for validation. Train set@ 70 per cent & random state to run same analysis later

x_train1, x_test1, y_train1, y_test1 = train_test_split(x1, y1, test_size=0.3, random_state=0)

#Fit the created object(linear regression) and apply it to split set, as with sklearn. score is r-squared

reg = LinearRegression().fit(x_train1, y_train1)

reg.score(x_train1, y_train1)

# Predicting the t & t case. How model did on the data we trained it on & how it extrapolates to a test case

reg.coef_

y_hat1_train= reg.predict(x_train1)

plt.scatter(x_train1, y_train1)
plt.scatter(x_train1, y_hat1_train)
plt.show()

----# Extrapolating the trained model on test

y_hat1_test= reg.predict(x_test1)
plt.scatter(x_test1, y_test1)
plt.scatter(x_test1, y_hat1_test)

plt.show()

#Studying other decision criteria- Mean Absolute Error(MAE),Root Mean Squared Error(RMSE)

MAE= mean_absolute_error(y_test1, y_hat1_test)

RMSE = math.sqrt(mean_absolute_error(y_test1, y_hat1_test))

print('Root Mean Squared Error=', RMSE)

print('Mean Absolute Error=', MAE)

# Using the Statsmodels OLS, as alternative to sklearn LR. Unlike in sklearn,Statsmodels requires adding a constant

import statsmodels.api as sm

df_final_vars['constant']= 1

x1b= df_final_vars[['constant', 'odometer']]

y1b= df_final_vars.price.values

x_train1b, x_test1b, y_train1b, y_test1b = train_test_split(x1b, y1b, test_size=0.3, random_state=0)

reg_sm1b= sm.OLS(y_train1b, x_train1b).fit()

reg_sm1b.summary()

# Specifying and estimating a multiple linear regression with the Statsmodel 

from statsmodels.stats.outliers_influence import variance_inflation_factor

x2 = df_final_vars[['constant', 'age', 'odometer', 'cylinders', 'text_len']]

y2= df_final_vars.price.values

x_train2, x_test2, y_train2, y_test2 = train_test_split(x2, y2, test_size=0.3, random_state=0)

reg_sm2= sm.OLS(y_train2, x_train2).fit()

reg_sm2.summary()

# Checking for multicollinearity between the ind. vars. using the variance-inflation-factor (VIF)

pd.Series([variance_inflation_factor(x2.values,i) for i in range(x2.shape[1])],index= x2.columns)

# For the 3rd model specification, which includes the categorical variables

x3= pd.get_dummies(df_final_vars[['constant','age','odometer','cylinders','condition','fuel','vin','type','text_len']])

y3= df_final_vars.price.values

x_train3, x_test3, y_train3, y_test3 = train_test_split(x3, y3, test_size=0.3, random_state=0)

reg_sm3= sm.OLS(y_train3, x_train3).fit()

reg_sm3.summary()

y_hat3= reg_sm3.predict(x_test3)

RMSE3 = math.sqrt(mean_squared_error(y_hat3, y_test3))

plt.scatter(y_hat3,y_test3)

# Validating the model using 5-fold cross validation to incorporate some randomness
# This involves 5 runs across the data as sanity check on the representativeness of the train-test split 

from sklearn.model_selection import cross_val_score

x4 = pd.get_dummies(df_final_vars[['age','odometer','cylinders','condition','fuel','vin','type','text_len']])

y4= df_final_vars.price.values

x_train4, x_test4, y_train4, y_test4 = train_test_split(x4, y4, test_size=0.3, random_state=0)

reg4 = LinearRegression().fit(x_train4, y_train4)

reg4.score(x_train4, y_train4)


# We may not have to add the train set. The cross_val will train the model in 5 runs

reg4 = LinearRegression()

scores = cross_val_score(reg4,x4,y4, cv=5, scoring= 'neg_mean-squared-error')
np.sqrt(np.abs(scores))

































                            





