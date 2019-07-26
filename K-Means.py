#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.cluster import KMeans
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


x1 = np.array([3, 1, 1, 2, 1, 6, 6, 6, 5, 6, 7, 8, 9, 8, 9, 9, 8, 2, 8, 5, 2, 7, 7, 6, 6])
x2 = np.array([5, 4, 6, 6, 5, 8, 6, 7, 6, 7, 1, 2, 1, 2, 3, 2, 3, 4, 1, 7, 5, 6, 7, 5, 3])


# In[3]:


plt.plot()
plt.xlim([0, 10])
plt.ylim([0, 10])
plt.title('Dataset')
plt.scatter(x1, x2)
plt.show()


# In[4]:


X = np.array(list(zip(x1, x2))).reshape(len(x1), 2)

colors = ['r', 'g', 'b']
markers = ['o', 'v', 's']

K = 3
k_means_model = KMeans(n_clusters=K).fit(X)

plt.plot()
plt.xlim([0, 10])
plt.ylim([0, 10])
for i, l in enumerate(k_means_model.labels_):
    plt.plot(x1[i], x2[i], color=colors[l], marker=markers[l],ls='None')

plt.show()

