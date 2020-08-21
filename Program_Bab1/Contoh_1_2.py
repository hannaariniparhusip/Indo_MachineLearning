#!/usr/bin/env python
# coding: utf-8

# In[3]:


#https://towardsdatascience.com/machine-learning-workflow-on-diabetes-data-part-01-573864fcc6b8
get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
diabetes= pd.read_csv('diabetesIndo.csv')
diabetes.columns


# In[4]:


diabetes.head()


# In[5]:


print("Dimensi data diabetes : {}".format(diabetes.shape))


# In[6]:


#Untuk mengidentifikasi berapa diabetes (1 diabetes dan 0 jk tidak)
diabetes.groupby('Luaran').size()


# In[7]:


diabetes.groupby('Luaran').hist(figsize=(9, 9))


# In[8]:


#mencermati data hilang
diabetes.isnull().sum()
diabetes.isna().sum()


# In[9]:


print("Total : ", diabetes[diabetes.Tekanandarah == 0].shape[0])


# In[10]:


#mencermati glukosa : jika nol berarti salah, karena tidak mungkin nol
print("Total : ", diabetes[diabetes.Glukosa == 0].shape[0])


# In[11]:


#teridentifikasi diabetes (berlabel 1) dan yang tidak diabetes (berlabel 0)
print(diabetes[diabetes.Glukosa == 0].groupby('Luaran')['Umur'].count())


# In[12]:


print("Total : ", diabetes[diabetes.Tebalkulit == 0].shape[0])
print(diabetes[diabetes.Tebalkulit == 0].groupby('Luaran')['Umur'].count())


# In[13]:


#BMI tidak boleh lebih kecil dari nol
print("Total : ", diabetes[diabetes.BMI == 0].shape[0])
print(diabetes[diabetes.BMI == 0].groupby('Luaran')['Umur'].count())


# In[14]:


print("Total : ", diabetes[diabetes.Insulin == 0].shape[0])
print(diabetes[diabetes.Insulin == 0].groupby('Luaran')['Umur'].count())


# In[16]:


#dihapus yang bernilai 0
diabetes_mod = diabetes[(diabetes.Tekanandarah != 0) & (diabetes.BMI != 0) & (diabetes.Glukosa != 0)]
print(diabetes_mod.shape)


# In[17]:


namafaktor = ['Kehamilan', 'Glukosa', 'Tekanandarah', 'Tebalkulit', 'Insulin', 'BMI', 'FungsiPedigree', 'Umur']
X = diabetes_mod[namafaktor]
y = diabetes_mod.Luaran


# In[18]:


#Mulai menganalisa dengan algoritma ML
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier


# In[20]:


models = []
models.append(('KNN', KNeighborsClassifier()))
models.append(('SVC', SVC()))
models.append(('LR', LogisticRegression()))
models.append(('DT', DecisionTreeClassifier()))
models.append(('GNB', GaussianNB()))
models.append(('RF', RandomForestClassifier()))
models.append(('GB', GradientBoostingClassifier()))

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold # ini ditambahkan krn tidk ada tetapi perlu


# In[21]:


X_train, X_test, y_train, y_test = train_test_split(X, y, stratify = diabetes_mod.Luaran, random_state=0)


# In[23]:


names = []
scores = []
for name, model in models:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    scores.append(accuracy_score(y_test, y_pred))
    names.append(name)
tr_split = pd.DataFrame({'Nama': names, 'Nilai': scores})
print(tr_split)


# In[26]:


#Cross validation
names = []
scores = []
for name, model in models:  
    kfold = KFold(n_splits=10, random_state=10) 
    score = cross_val_score(model, X, y, cv=kfold, scoring='0.1').mean()
    names.append(name)
    scores.append(score)
kf_cross_val = pd.DataFrame({'Nama': names, 'Nilai': scores})
print(kf_cross_val)


# In[27]:


names = []
scores = []
for name, model in models:  
    kfold = KFold(n_splits=10, random_state=10) 
    score = cross_val_score(model, X, y, cv=kfold, scoring='accuracy').mean()
    names.append(name)
    scores.append(score)
kf_cross_val = pd.DataFrame({'Nama': names, 'Nilai': scores})
print(kf_cross_val)


# In[29]:


#Menggunakan data yang sudah bersih untuk proses lebih lanjut
#https://towardsdatascience.com/machine-learning-workflow-on-diabetes-data-part-01-573864fcc6b8
get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
diabetes = pd.read_csv('diabetesIndo.csv')
diabetes.columns


# In[30]:


diabetes_mod = diabetes[(diabetes.Tekanandarah != 0) & (diabetes.BMI != 0) & (diabetes.Glukosa != 0)]
print(diabetes_mod.shape)


# In[31]:


#menyimpan data pada file Simpan. csv
diabetes_mod.to_csv('Cleandiabetes.csv')


# In[ ]:




