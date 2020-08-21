#!/usr/bin/env python
# coding: utf-8

# In[13]:


import numpy as np 
import matplotlib.pyplot as plt
list = [[1,2],[4,3],[5, 6],[8, 7],[9, 10],[12, 11],[13,14],[16, 15],[17,18],[20, 19]] 
x, y = zip(*list)
plt.plot(x,y, linestyle = '',color='darkblue',marker = '*',markersize=10,markeredgecolor='darkblue',markerfacecolor='purple' )
plt.xlabel("x")
plt.ylabel("y")
plt.show()


# In[14]:


from pandas import DataFrame
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
data = DataFrame(list,columns=['x','y'])
print (data)
kmeans = KMeans(n_clusters=2).fit(data)
centroids = kmeans.cluster_centers_
print(centroids)
plt.plot(x,y, linestyle = '',color='darkblue',marker = '*',markersize=10,markeredgecolor='darkblue',markerfacecolor='purple' )
plt.scatter(centroids[:, 0], centroids[:, 1], c='green', s=90)
plt.xlabel("x")
plt.ylabel("y")
plt.show()


# In[ ]:





# In[ ]:




