#!/usr/bin/env python
# coding: utf-8

# In[3]:


#https://machinelearningmastery.com/tutorial-to-implement-k-nearest-neighbors-in-python-from-scratch/
#fungsi untuk menghitung jarak euclide antar 2 vektor 
def euclidean_distance(row1, row2):
	distance = 0.0
	for i in range(len(row1)-1):
		distance += (row1[i] - row2[i])**2
	return sqrt(distance)


# In[10]:


# hitung jarak Euclide dari data
from math import sqrt
 
# fungsi untuk menghitung jarak euclide antar 2 vektor
def euclidean_jarak(baris1, baris2):
	jarak = 0.0
	for i in range(len(baris1)-1):
		jarak += (baris1[i] - baris2[i])**2
	return sqrt(jarak)
 
# Data untuk menguji fungsi jarak 
dataset = [[2.7810836,2.550537003,0],
	[1.465489372,2.362125076,0],
	[3.396561688,4.400293529,0],
	[1.38807019,1.850220317,0],
	[3.06407232,3.005305973,0],
	[7.627531214,2.759262235,1],
	[5.332441248,2.088626775,1],
	[6.922596716,1.77106367,1],
	[8.675418651,-0.242068655,1],
	[7.673756466,3.508563011,1]]
baris0 = dataset[0]
for row in dataset:
	jarak = euclidean_jarak(baris0, row)
	print(jarak)


# In[12]:


# Kita kumpulkan yang paling dekat
def get_neighbors(train, test_row, num_neighbors):
	distances = list()
	for train_row in train:
		dist = euclidean_jarak(test_row, train_row)
		distances.append((train_row, dist))
	distances.sort(key=lambda tup: tup[1])
	neighbors = list()
	for i in range(num_neighbors):
		neighbors.append(distances[i][0])
	return neighbors


# In[13]:



# Fungsi jarak diuji
dataset = [[2.7810836,2.550537003,0],
	[1.465489372,2.362125076,0],
	[3.396561688,4.400293529,0],
	[1.38807019,1.850220317,0],
	[3.06407232,3.005305973,0],
	[7.627531214,2.759262235,1],
	[5.332441248,2.088626775,1],
	[6.922596716,1.77106367,1],
	[8.675418651,-0.242068655,1],
	[7.673756466,3.508563011,1]]
neighbors = get_neighbors(dataset, dataset[0], 3)
for neighbor in neighbors:
	print(neighbor)


# In[14]:


# membuat dugaan 
#Membuat pengelompokan bersasarkan hasil
def predict_classification(train, test_row, num_neighbors):
	neighbors = get_neighbors(train, test_row, num_neighbors)
	output_values = [row[-1] for row in neighbors]
	dugaan = max(set(output_values), key=output_values.count)
	return dugaan


# In[16]:


# Data yang diuji
dataset = [[2.7810836,2.550537003,0],
	[1.465489372,2.362125076,0],
	[3.396561688,4.400293529,0],
	[1.38807019,1.850220317,0],
	[3.06407232,3.005305973,0],
	[7.627531214,2.759262235,1],
	[5.332441248,2.088626775,1],
	[6.922596716,1.77106367,1],
	[8.675418651,-0.242068655,1],
	[7.673756466,3.508563011,1]]
dugaan = predict_classification(dataset, dataset[0], 3)
print('Yang diharapkan %d, Diperoleh %d.' % (dataset[0][-1], dugaan))


# In[ ]:




