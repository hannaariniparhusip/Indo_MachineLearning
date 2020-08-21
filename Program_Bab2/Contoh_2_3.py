#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

df = pd.DataFrame({
'x': [1, 5, 7, 6.5, 9.5, 13.75, 17.15, 14, 12, 16],
'y':[3, 2, 4, 1.5, 6.25, 8.5, 11.25, 10.6, 8, 19.5]})

np.random.seed(200)
k = 3
# pusat[i] = [x, y]
pusat = {i+1: [np.random.randint(0, 25), np.random.randint(0, 25)]for i in range(k)}
fig = plt.figure
plt.scatter(df['x'], df['y'], color='k')
colmap = {1: 'r', 2: 'g', 3: 'b'} 
for i in pusat.keys():
    plt.scatter(*pusat[i], color=colmap[i]) 
plt.xlim(0, 20)
plt.ylim(0, 20)
plt.show()
 


# In[ ]:





# In[10]:


# Inisialisasi
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

df = pd.DataFrame({
'x': [1, 5, 7, 6.5, 9.5, 13.75, 17.15, 14, 12, 16],
'y':[3, 2, 4, 1.5, 6.25, 8.5, 11.25, 10.6, 8, 19.5]})

np.random.seed(200)
k = 3
# pusat[i] = [x, y]
pusat = {i+1: [np.random.randint(0, 25), np.random.randint(0, 25)]for i in range(k)}
fig = plt.figure
plt.scatter(df['x'], df['y'], color='k',)
colmap = {1: 'r', 2: 'r', 3: 'r'} 
for i in pusat.keys():
    plt.scatter(*pusat[i], color=colmap[i], marker='*',s=50) 
plt.xlim(0, 20)
plt.ylim(0, 20)
plt.show()
 


# In[12]:


## Penugasan 

def penugasan(df, pusat): 
    for i in pusat.keys():
     # sqrt((x1 - x2)^2 - (y1 - y2)^2)
       df['jarakdari_{}'.format(i)] = (
        np.sqrt((df['x'] - pusat[i][0]) ** 2+ (df['y'] - pusat[i][1]) ** 2))
    pusat_jarak_cols = ['jarakdari_{}'.format(i) for i in pusat.keys()] 
    df['terdekat'] = df.loc[:, pusat_jarak_cols].idxmin(axis=1)
    df['terdekat'] = df['terdekat'].map(lambda x: int(x.lstrip('jarakdari_')))
    df['warna'] = df['terdekat'].map(lambda x: colmap[x]) 
    return df
df = penugasan(df, pusat) 
print(df.head())
fig = plt.figure
plt.scatter(df['x'], df['y'], color=df['warna'], alpha=0.5, edgecolor='k')
for i in pusat.keys():
    plt.scatter(*pusat[i], color=colmap[i],marker='*')
plt.xlim(0, 20)
plt.ylim(0, 20) 
plt.show()


# In[15]:


import copy
pusat_lama = copy.deepcopy(pusat)

def update(k):
    for i in pusat.keys():
        pusat[i][0] = np.mean(df[df['terdekat'] == i]['x'])
        pusat[i][1] = np.mean(df[df['terdekat'] == i]['y']) 
    return k
pusat = update(pusat) 
fig = plt.figure
ax = plt.axes()
plt.scatter(df['x'], df['y'], color=df['warna'], alpha=0.5, edgecolor='k') 
for i in pusat.keys():
    plt.scatter(*pusat[i], color=colmap[i],marker='*',s=50)
plt.xlim(0, 20)
plt.ylim(0, 20)
for i in pusat_lama.keys(): 
    x_lama = pusat_lama[i][0] 
    y_lama = pusat_lama[i][1]
    dx = (pusat[i][0] - pusat_lama[i][0]) * 0.75 
    dy = (pusat[i][1] - pusat_lama[i][1]) * 0.75
plt.show()



# In[17]:


## Pengulangan penugasan
df = penugasan(df, pusat) # Plot results
fig = plt.figure
plt.scatter(df['x'], df['y'], color=df['warna'], alpha=0.5, edgecolor='k') 
for i in pusat.keys():
    plt.scatter(*pusat[i], color=colmap[i],marker='*',s=50) 
plt.xlim(0, 20)
plt.ylim(0, 20)
plt.show()


# In[18]:



# Dilanjutkan hingga pusat tidak berubah
while True:
    pusat_terdekat = df['terdekat'].copy(deep=True)
    pusat = update(pusat) 
    df = penugasan(df, pusat)
    if pusat_terdekat.equals(df['terdekat']):
        break
    
fig = plt.figure
plt.scatter(df['x'], df['y'], color=df['warna'], alpha=0.5, edgecolor='k') 
for i in pusat.keys(): 
    plt.scatter(*pusat[i], color=colmap[i],marker='*',s=50)
plt.xlim(0, 20)
plt.ylim(0, 20) 
plt.show()


# In[ ]:




