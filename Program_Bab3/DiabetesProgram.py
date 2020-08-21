
# coding: utf-8

# In[1]:


#https://towardsdatascience.com/machine-learning-workflow-on-diabetes-data-part-01-573864fcc6b8
get_ipython().magic(u'matplotlib inline')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
diabetes = pd.read_csv('E:/JAN2016/KULIAH20182019/SEM2_20182019/ANALISA_DATA/PROJECTAnalisaData/diabetesIndo.csv')
diabetes.columns 


# In[3]:


diabetes.head()


# In[4]:


print("Dimensi data diabetes : {}".format(diabetes.shape))


# In[5]:


#Untuk mengidentifikasi berapa diabetes (1 di outcomes dan 0 jk tidak)
diabetes.groupby('Luaran').size()


# In[6]:


diabetes.groupby('Luaran').hist(figsize=(9, 9))


# In[7]:


#Untuk mengenali ada tidaknya missing data, ternyata tidak
diabetes.isnull().sum()
diabetes.isna().sum()


# In[9]:


print("Total : ", diabetes[diabetes.Tekanandarah == 0].shape[0])
print(diabetes[diabetes.Tekanandarah == 0].groupby('Luaran')['Umur'].count())


# In[10]:


#Mengetahui banyaknya orang dengan level glukosa 0 seharusnya tidak boleh ada, tetapi terbaaca ada 5
print("Total : ", diabetes[diabetes.Glukosa == 0].shape[0])


# In[11]:


print(diabetes[diabetes.Glukosa == 0].groupby('Luaran')['Umur'].count())


# In[12]:


print("Total : ", diabetes[diabetes.Tebalkulit == 0].shape[0])

print(diabetes[diabetes.Tebalkulit == 0].groupby('Luaran')['Umur'].count())


# In[13]:


#BMI tidak boleh lebih kecil dari nol
print("Total : ", diabetes[diabetes.BMI == 0].shape[0])



# In[14]:


print(diabetes[diabetes.BMI == 0].groupby('Luaran')['Umur'].count())


# In[15]:


print("Total : ", diabetes[diabetes.Insulin == 0].shape[0])
print(diabetes[diabetes.Insulin == 0].groupby('Luaran')['Umur'].count())


# In[17]:


diabetes_mod = diabetes[(diabetes.Tekanandarah != 0) & (diabetes.BMI != 0) & (diabetes.Glukosa != 0)]
print(diabetes_mod.shape)


# In[18]:


namafaktor = ['Kehamilan', 'Glukosa', 'Tekanandarah', 'Tebalkulit', 'Insulin', 'BMI', 'FungsiPedigree', 'Umur']
X = diabetes_mod[namafaktor]
y = diabetes_mod.Luaran


# In[19]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier


# In[20]:


#Langkah 15
models = []
models.append(('KNN', KNeighborsClassifier()))
models.append(('SVC', SVC()))
models.append(('LR', LogisticRegression()))
models.append(('DT', DecisionTreeClassifier()))
models.append(('GNB', GaussianNB()))
models.append(('RF', RandomForestClassifier()))
models.append(('GB', GradientBoostingClassifier()))


# In[21]:


#Langkah 16
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold # ini ditambahkan krn tidk ada tetapi perlu


# In[22]:


X_train, X_test, y_train, y_test = train_test_split(X, y, stratify = diabetes_mod.Luaran, random_state=0)


# In[ ]:


names = []
scores = []
for name, model in models:
    model.fit(X_train, y_train)
    y_prediksi = model.predict(X_test)
    scores.append(accuracy_score(y_test, y_prediksi))
    names.append(names)
tr_split = pd.DataFrame({'Nama': names, 'Nilai': scores})
print(tr_split)


# In[ ]:



names = []
scores = []
for name, model in models:  
    kfold = KFold(n_splits=10, random_state=10) 
    score = cross_val_score(model, X, y, cv=kfold, scoring='accuracy').mean()
    names.append(name)
    scores.append(score)
kf_cross_val = pd.DataFrame({'Nama': names, 'Nilai': scores})
print(kf_cross_val)


# In[95]:


# plot nilai akurasi dengan seaborn


axis = sns.barplot(x = 'Nama', y = 'Nilai', data = kf_cross_val)
axis.set(xlabel='Klasifikasi', ylabel='Akurasi')
for p in axis.patches:
    tinggi = p.get_height()
    axis.text(p.get_x() + p.get_width()/2, tinggi + 0.005, '{:1.4f}'.format(tinggi), ha="center") 
    
plt.show()

