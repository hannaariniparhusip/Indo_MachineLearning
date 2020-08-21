#!/usr/bin/env python
# coding: utf-8

# In[6]:


#Clustering untuk bentuk batas komples
from sklearn.datasets import make_moons
X, y = make_moons(200, noise=.05, random_state=0)


# In[7]:


import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
labels = KMeans(2, random_state=0).fit_predict(X)
plt.scatter(X[:, 0], X[:, 1], c=labels,s=50, cmap='viridis');


# In[8]:


from sklearn.cluster import SpectralClustering
model = SpectralClustering(n_clusters=2, affinity='nearest_neighbors',
                           assign_labels='kmeans')
labels = model.fit_predict(X)
plt.scatter(X[:, 0], X[:, 1], c=labels,
            s=50, cmap='viridis');


# In[ ]:





# In[ ]:




