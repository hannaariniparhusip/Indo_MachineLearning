#!/usr/bin/env python
# coding: utf-8

# In[2]:


#https://github.com/lucko515/breast-cancer-classification/blob/master/NaiveBayesClassifier.ipynb
#importing libs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


#Untuk mengajarkan bagaimana menghitung jarak antar data
X = np.array([[1, 3],[4, 6],[3, 2],[7, 5],[7, 6]])
for i in range(len(X)):
    plt.scatter(X[i][1], X[i][0])

plt.scatter(3, 4, color='red')
circle = plt.Circle((3, 4), radius=2, alpha=0.4)
plt.gca().add_patch(circle)
plt.axis('scaled')
plt.show()

jarakEuclid = np.sqrt((3-2)**2 + (4-3)**2)
print(jarakEuclid )


# In[5]:


class NaiveBayesClassifier(object):
    def __init__(self):
        pass
    
    #Input: X - faktor dalam data training
    #       y - Label dalam data trainiing
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
        
        self.no_of_classes = np.max(self.y_train) + 1
       
      
    # fungsi untuk menghitung semua titi/contoh data dalam radius yang dipenuhi
    def euclidianDistance(self, Xtest, Xtrain):
        return np.sqrt(np.sum(np.power((Xtest - Xtrain), 2)))
    
   
    # Ini adalah fungsi utama untuk melakukan prediksi 
    # semua perhitungan menggunakan contoh data uji yang baru
   # Terdapat 4 tahap untuk dilakukan
   # 1. Hitung Probabilitas prior, contoh P(A) =banyaknya elemen dalam 1 klas/ total sampel 
    #.2 Hitung probabilitas Margin P(X) = banyaknya elemen dalam radius(lingkaran) / total sampel 
    # 3. Hitung likelihood (P(X|A) = banyaknya elemen dalam klas yang dihitung/ total sampel 
    # 4. Hitung probabilitas posterior =P(A|X) = (P(X|A) * P(A)) / P(X)
    #Kerjakan semua tahap itu untuk semua klas dalam himpunan data
    
    # 
    #Input: X - himpunan data uji
    #      
    # radius = parameter yangmenyatakan seberapa besar lingkaran untuk data yang baru, default=2
    
    def predict(self, X, radius=0.4):   
        pred = []
        
        #Membuat list/daftar banyaknya element untuk setiap klass dalam datalatih (trainset)
        members_of_class = []
        for i in range(self.no_of_classes):
            counter = 0
            for j in range(len(self.y_train)):
                if self.y_train[j] == i:
                    counter += 1
            members_of_class.append(counter)
        
        #Memasuki proses dalam memprediksi
        for t in range(len(X)):
            #Membuat list kosong untuk setiap probabilitas klas
            prob_of_classes = [] 
            #Looping pada setiap klass dalam data
            for i in range(self.no_of_classes):
                
                #1. step : probabilitas prior P(class)=banyaknya elemen dalam klas/ total elemen
                prior_prob = members_of_class[i]/len(self.y_train)

                #Step : Probabilitas margin P(X) = banyaknya elemen dalam radius(lingkaran)/banyaknya elemen
                # Catatan : Pada looping informasi yang sama untuk step 3 
                inRadius_no = 0
                # Menghitung banyaknya titik dalam klas yang dicermati dalam lingkaran
                inRadius_no_current_class = 0
                
                for j in range(len(self.X_train)):
                    if self.euclidianDistance(X[t], self.X_train[j]) < radius:
                        inRadius_no += 1
                        if self.y_train[j] == i:
                            inRadius_no_current_class += 1
                
                #Menghitung probabilitas margin 
                margin_prob = inRadius_no/len(self.X_train)
                 
                #3. step : likelihood P(X|currect_class) - banyaknya element dalam klas sekarang/total element
                likelihood = inRadius_no_current_class/len(self.X_train)
                
                # 4. step : Probabilitas posterioe formula dari teorema Bayes 
                 #  P(currect_class|X)= (likelihood*prior_prob)/margin_prob
                    
                post_prob = (likelihood * prior_prob)/margin_prob
                prob_of_classes.append(post_prob)
            
       # dapatkan indeks dengan element terbesar( klass dengan probabilitas terbesar)
            pred.append(np.argmax(prob_of_classes))
               
        return pred

#Tahap 3. Menyusun fungsi untuk menguji akurasi
#Lanjutkan pada file di atas 
def akurasi(y_tes, y_pred):
    benar = 0
    for i in range(len(y_pred)):
        if(y_tes[i] == y_pred[i]):
            benar += 1
    return (benar/len(y_tes))*100


# In[6]:


#Kita ingin mempelajari data file di atas (karena dalam fungsi run, kita tidak bisa panggil, 
#sehinggal diulang panggil)
dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:, [2, 3]].values
y = dataset.iloc[:, 4].values
dataset.columns 
dataset.head()
print("Dimensi data  : {}".format(dataset.shape))




# In[7]:


def run():
    # Data dicoba
    dataset = pd.read_csv('Social_Network_Ads.csv')
    X = dataset.iloc[:, [2, 3]].values
    y = dataset.iloc[:, 4].values
        

    #Pemisahan data menjadi data latih (training set dan data uji (test set))
   # from sklearn.model_selection import model_selection
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

    # Penyekalaan faktor-faktor
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    
    #Menguji  Naive Bayes Classifier yang dibuat
    NB = NaiveBayesClassifier()
    NB.fit(X_train, y_train)
    
    y_pred = NB.predict(X_test, radius=0.4)
    
    #sklearn
    from sklearn.naive_bayes import GaussianNB
    NB_sk = GaussianNB()
    NB_sk.fit(X_train, y_train)
    
    sk_pred = NB_sk.predict(X_test)
     
    
    print("Akurasi untuk NBC disini adalah: ", akurasi(y_test, y_pred), "%")
    print("Akurasi NBC dari sklearn: ",akurasi(y_test, sk_pred), "%")

run()


# In[ ]:




