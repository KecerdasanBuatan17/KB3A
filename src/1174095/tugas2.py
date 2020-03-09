# -*- coding: utf-8 -*-
"""
Created on Sun Mar  8 11:05:56 2020

@author: dzihan
"""

# In[1]:

# load dataset (menggunakan student-mat.csv)
import pandas as pd #mengimport library pandas sebagai pd
sate = pd.read_csv('dataset/student-mat.csv', sep=';') #variabel sate berfungsi untuk read file student-mat.csv
len(sate) #mengetahui jumlah baris pada data yang dipanggil


# In[2]:

# generate binary label (pass/fail) based on G1+G2+G3 (test grades, each 0-20 pts); threshold for passing is sum>=30
sate['pass'] = sate.apply(lambda row: 1 if (row['G1']+row['G2']+row['G3']) >= 35 else 0, axis=1) #mendeklarasikan pass/fail nya data berdasarkan G1+G2+G3.
sate = sate.drop(['G1', 'G2', 'G3'], axis=1) #untuk mengetahui baris G1+G2+G3 ditambahkan, dan hasilnya sama dengan 35 maka axisnya 1.
sate.head() #memanggil variabel sate dengan ketentuan head ini digunakan untuk mengembalikan baris n atas 5 secara default dari frame atau seri data


# In[3]:

# use one-hot encoding on categorical columns
sate = pd.get_dummies(sate, columns=['sex', 'school', 'address', 'famsize', 'Pstatus', 'Mjob', 'Fjob', 
                               'reason', 'guardian', 'schoolsup', 'famsup', 'paid', 'activities',
                               'nursery', 'higher', 'internet', 'romantic'])  #variabel medan dikonversi menjadi bentuk yang lebih baik dalam prediksi dan memanggil seluruh atribut 
sate.head() #memanggil variabel sate dengan ketentuan head ini digunakan untuk mengembalikan baris n atas 5 secara default dari frame atau seri data


# In[4]:

# shuffle rows
sate = sate.sample(frac=1) #mengembalikan variabel sate menjadi sampel acak dengan frac=1
# split training and testing data
sate_train = sate[:500] #membuat variabel baru sate_train
sate_test = sate[500:] #membuat variabel baru sate_test yang sisa dari train

sate_train_att = sate_train.drop(['pass'], axis=1) #membuat variabel baru dengan ketentuan dari sate_train
sate_train_pass = sate_train['pass'] #membuat variabel baru dengan ketentuan dari sate_train

sate_test_att = sate_test.drop(['pass'], axis=1) #membuat variabel baru dengan ketentuan dari sate_test
sate_test_pass = sate_test['pass'] #membuat variabel baru dengan ketentuan dari sate_test

sate_att = sate.drop(['pass'], axis=1)  #membuat variabel sate_att sebagai salinan dari sate
sate_pass = sate['pass'] #membuat variabel sate_pass sebagai salinan dari sate

# number of passing students in whole dataset:
import numpy as np #mengimport module numpy sebagai np y 
print("Passing: %d out of %d (%.2f%%)" % (np.sum(sate_pass), len(sate_pass), 100*float(np.sum(sate_pass)) / len(sate_pass))) #untuk mengembalikan nilai passing dari pelajar dari keseluruhan dataset dengan cara print.


# In[5]:

# fit a decision tree 
from sklearn import tree #import tree dari library sklearn
martabak = tree.DecisionTreeClassifier(criterion="entropy", max_depth=5) #membuat variabel asahan sebagai decisiontree, dengan criterion fungsi mengukur kualitas split
martabak = martabak.fit(sate_train_att, sate_train_pass) #training varibael asahan dengan data dari variabel sate.


# In[6]:

# visualize tree
import graphviz #import library graphviz sebagai perangkat lunak visualisasi grafik open source
dot_data = tree.export_graphviz(martabak, out_file=None, label="all", impurity=False, proportion=True,
                                feature_names=list(sate_train_att), class_names=["fail", "pass"], 
                                filled=True, rounded=True) #mengambil data untuk diterjemahkan ke grafik
graph = graphviz.Source(dot_data) #membuat variabel graph sebagai grafik yang di ambil dari dot_data
graph #memanggil graph


# In[7]:

# save tree
tree.export_graphviz(martabak, out_file="student-performance.dot", label="all", impurity=False, proportion=True,
                     feature_names=list(sate_train_att), class_names=["fail", "pass"], 
                     filled=True, rounded=True) #save tree sebagai export graphviz ke file student-performance.dot


# In[8]:

#asahan.score(medan_test_att, medan_test_pass)

martabak.score(sate_att, sate_pass) #score juga disebut prediksi dengan diberi beberapa data input baru


# In[9]:

from sklearn.model_selection import cross_val_score #import class cross_val_score dari sklearn
scores = cross_val_score(martabak, sate_att, sate_pass, cv=5) #mengevaluasi score dengan validasi silang
# show average score and +/- two standard deviations away (covering 95% of scores)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2)) #print akurasi 


# In[10]:

for max_depth in range(1, 20):
    martabak = tree.DecisionTreeClassifier(criterion="entropy", max_depth=max_depth)
    scores = cross_val_score(martabak, sate_att, sate_pass, cv=5)
    print("Max depth: %d, Accuracy: %0.2f (+/- %0.2f)" % (max_depth, scores.mean(), scores.std() * 2)) 
    
    #Disini ini menunjukkan seberapa dalam di tree itu. Semakin dalam tree, semakin banyak perpecahan yang dimilikinya dan menangkap lebih banyak informasi tentang data.


# In[11]:

depth_acc = np.empty((19,3), float) #Dengan 19 sebagai bentuk array kosong, 3 sebagai output data-type
cimin = 0 #variabel jakarta sebagai array 0 
for max_depth in range(1, 20): #perulangan dengan max_depth
    martabak = tree.DecisionTreeClassifier(criterion="entropy", max_depth=max_depth) #variabel asahan untuk decision tree dengan ketentuan entropy
    scores = cross_val_score(martabak, sate_att, sate_pass, cv=5) #scores diambil dari data cross_val_score
    depth_acc[cimin,0] = max_depth #mengembalikan array dengan ketentuan 0 dan max_depth
    depth_acc[cimin,1] = scores.mean() #mengembalikan array dengan ketentuan 1 dan scores.mean
    depth_acc[cimin,2] = scores.std() * 2 #mengembalikan array dengan ketentuan 2 dan scores.std, std berarti menghitung standar deviasi 
    cimin += 1
    
depth_acc #Depth acc akan membuat array kosong dengan mengembalikan array baru dengan bentuk dan tipe yang diberikan


# In[12]:

import matplotlib.pyplot as plt #import matplotlip sebagai plt
fig, ax = plt.subplots() #fig dan ax menggunakan subplots untuk membuat gambar
ax.errorbar(depth_acc[:,0], depth_acc[:,1], yerr=depth_acc[:,2]) #membuat error bar kemudian grafik akan ditampilkan menggunakan show
plt.show() #menampilkan plot dari data yang ada