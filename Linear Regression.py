#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


companies = pd.read_csv('1000_Companies.csv')
companies.head()


# In[3]:


X = companies.iloc[:,:-1].values
y = companies.iloc[:,4].values


# In[4]:


sns.heatmap(companies.corr())


# In[5]:


labelencoder = LabelEncoder()
X[:,3] = labelencoder.fit_transform(X[:,3])

onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()

X = X[:,1:]

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=0)


# In[6]:


lin_reg = LinearRegression()
lin_reg.fit(X_train,y_train)


# In[7]:


y_pred = lin_reg.predict(X_test)


# In[8]:


print(lin_reg.coef_)


# In[9]:


print(lin_reg.intercept_)


# In[10]:


r2_score(y_test,y_pred)

