#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Importing required libraries.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


#Reading our input data for House Price Prediction.
customers = pd.read_csv(r'C:\housing Data\housing_data.csv')
customers.head()


# In[3]:


#Describing our data.
customers.describe()


# In[4]:


#Analyzing information from our data.
customers.info()


# In[5]:


#Plots to visualize data of House Price Prediction.
sns.pairplot(customers)


# In[6]:


#Scaling our data.
scaler = StandardScaler()
X=customers.drop(['Price','Address'],axis=1)
y=customers['Price']
cols = X.columns
X = scaler.fit_transform(X)


# In[7]:


#Splitting our data for train and test purposes.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)


# In[8]:


#Training our Linear Regression model for House Price Prediction.
lr = LinearRegression()
lr.fit(X_train,y_train)
pred = lr.predict(X_test)
r2_score(y_test,pred)


# In[9]:


#Lets visualize our predictions of House Price Prediction.
sns.scatterplot(x=y_test, y=pred)


# In[10]:


# Plotting the residuals of our House Price Prediction model.
sns.histplot((y_test-pred),bins=50,kde=True)


# In[11]:


#Observing the coefficients.
cdf=pd.DataFrame(lr.coef_, cols, ['coefficients']).sort_values('coefficients',ascending=False)
cdf

