#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score


# In[2]:


dataset = pd.read_csv('diabetes.csv')
zero_not_accepted = ['Glucose', 'BloodPressure', 'SkinThickness', 'BMI', 'Insulin']

for column in zero_not_accepted:
    dataset[column] = dataset[column].replace(0,  np.NaN)
    mean = int(dataset[column].mean(skipna=True))
    dataset[column] = dataset[column].replace(np.NaN, mean)


# In[3]:


X = dataset.iloc[:, 0:8]
y = dataset.iloc[:, 8]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0, test_size = 0.2 )


# In[4]:


sc_X = StandardScaler()
X_train  = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)


# In[5]:


classifier = KNeighborsClassifier(n_neighbors = 11, p = 2, metric = 'euclidean' )
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
print(y_pred)


# In[6]:


cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix")
print(cm)


# In[7]:


print("F1 Score: ", f1_score(y_test,y_pred))
print("Accuracy Score: ", accuracy_score(y_test , y_pred))

