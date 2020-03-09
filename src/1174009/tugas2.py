# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 17:38:32 2020

@author: root
"""

# In[1]:

# load dataset (menggunakan student-mat.csv)
import pandas as pd #mengimport library pandas sebagai pd
karawang = pd.read_csv('dataset/student-mat.csv', sep=';') #variabel karawang berfungsi untuk read file student-mat.csv
len(karawang) #mengetahui jumlah baris pada data yang dipanggil


# In[2]:

# generate binary label (pass/fail) based on G1+G2+G3 (test grades, each 0-20 pts); threshold for passing is sum>=30
karawang['pass'] = karawang.apply(lambda row: 1 if (row['G1']+row['G2']+row['G3']) >= 35 else 0, axis=1) #mendeklarasikan pass/fail nya data berdasarkan G1+G2+G3.
karawang = karawang.drop(['G1', 'G2', 'G3'], axis=1) #untuk mengetahui baris G1+G2+G3 ditambahkan, dan hasilnya sama dengan 35 maka axisnya 1.
karawang.head() #memanggil variabel sate dengan ketentuan head ini digunakan untuk mengembalikan baris n atas 5 secara default dari frame atau seri data


# In[3]:

# use one-hot encoding on categorical columns
karawang = pd.get_dummies(karawang, columns=['sex', 'school', 'address', 'famsize', 'Pstatus', 'Mjob', 'Fjob', 
                               'reason', 'guardian', 'schoolsup', 'famsup', 'paid', 'activities',
                               'nursery', 'higher', 'internet', 'romantic'])  #variabel medan dikonversi menjadi bentuk yang lebih baik dalam prediksi dan memanggil seluruh atribut 
karawang.head() #memanggil variabel karawang dengan ketentuan head ini digunakan untuk mengembalikan baris n atas 5 secara default dari frame atau seri data


# In[4]:

# shuffle rows
karawang = karawang.sample(frac=1) #mengembalikan variabel karawang menjadi sampel acak dengan frac=1
# split training and testing data
karawang_train = karawang[:500] #membuat variabel baru karawang_train
karawang_test = karawang[500:] #membuat variabel baru karawang_test yang sisa dari train

karawang_train_att = karawang_train.drop(['pass'], axis=1) #membuat variabel baru dengan ketentuan dari karawang_train
karawang_train_pass = karawang_train['pass'] #membuat variabel baru dengan ketentuan dari karawang_train

karawang_test_att = karawang_test.drop(['pass'], axis=1) #membuat variabel baru dengan ketentuan dari karawang_test
karawang_test_pass = karawang_test['pass'] #membuat variabel baru dengan ketentuan dari karawang_test

karawang_att = karawang.drop(['pass'], axis=1)  #membuat variabel karawang_att sebagai salinan dari karawang
karawang_pass = karawang['pass'] #membuat variabel karawang_pass sebagai salinan dari karawang

# number of passing students in whole dataset:
import numpy as np #mengimport module numpy sebagai np y 
print("Passing: %d out of %d (%.2f%%)" % (np.sum(karawang_pass), len(karawang_pass), 100*float(np.sum(karawang_pass)) / len(karawang_pass))) #untuk mengembalikan nilai passing dari pelajar dari keseluruhan dataset dengan cara print.


# In[5]:

# fit a decision tree 
from sklearn import tree #import tree dari library sklearn
cikampek = tree.DecisionTreeClassifier(criterion="entropy", max_depth=5) #membuat variabel asahan sebagai decisiontree, dengan criterion fungsi mengukur kualitas split
cikampek = cikampek.fit(karawang_train_att, karawang_train_pass) #training varibael cikampek dengan data dari variabel karawang.


# In[6]:

# visualize tree
import graphviz #import library graphviz sebagai perangkat lunak visualisasi grafik open source
dot_data = tree.export_graphviz(cikampek, out_file=None, label="all", impurity=False, proportion=True,
                                feature_names=list(karawang_train_att), class_names=["fail", "pass"], 
                                filled=True, rounded=True) #mengambil data untuk diterjemahkan ke grafik
graph = graphviz.Source(dot_data) #membuat variabel graph sebagai grafik yang di ambil dari dot_data
graph #memanggil graph


# In[7]:

# save tree
tree.export_graphviz(cikampek, out_file="student-performance.dot", label="all", impurity=False, proportion=True,
                     feature_names=list(karawang_train_att), class_names=["fail", "pass"], 
                     filled=True, rounded=True) #save tree sebagai export graphviz ke file student-performance.dot


# In[8]:

#asahan.score(medan_test_att, medan_test_pass)

cikampek.score(karawang_att, karawang_pass) #score juga disebut prediksi dengan diberi beberapa data input baru


# In[9]:

from sklearn.model_selection import cross_val_score #import class cross_val_score dari sklearn
scores = cross_val_score(cikampek, karawang_att, karawang_pass, cv=5) #mengevaluasi score dengan validasi silang
# show average score and +/- two standard deviations away (covering 95% of scores)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2)) #print akurasi 


# In[10]:

for max_depth in range(1, 20):
    cikampek = tree.DecisionTreeClassifier(criterion="entropy", max_depth=max_depth)
    scores = cross_val_score(cikampek, karawang_att, karawang_pass, cv=5)
    print("Max depth: %d, Accuracy: %0.2f (+/- %0.2f)" % (max_depth, scores.mean(), scores.std() * 2)) 
    
    #Disini ini menunjukkan seberapa dalam di tree itu. Semakin dalam tree, semakin banyak perpecahan yang dimilikinya dan menangkap lebih banyak informasi tentang data.


# In[11]:

depth_acc = np.empty((19,3), float) #Dengan 19 sebagai bentuk array kosong, 3 sebagai output data-type
sadang = 0 #variabel sadang sebagai array 0 
for max_depth in range(1, 20): #perulangan dengan max_depth
    cikampek = tree.DecisionTreeClassifier(criterion="entropy", max_depth=max_depth) #variabel cikampek untuk decision tree dengan ketentuan entropy
    scores = cross_val_score(cikampek, karawang_att, karawang_pass, cv=5) #scores diambil dari data cross_val_score
    depth_acc[sadang,0] = max_depth #mengembalikan array dengan ketentuan 0 dan max_depth
    depth_acc[sadang,1] = scores.mean() #mengembalikan array dengan ketentuan 1 dan scores.mean
    depth_acc[sadang,2] = scores.std() * 2 #mengembalikan array dengan ketentuan 2 dan scores.std, std berarti menghitung standar deviasi 
    sadang += 1
    
depth_acc #Depth acc akan membuat array kosong dengan mengembalikan array baru dengan bentuk dan tipe yang diberikan


# In[12]:

import matplotlib.pyplot as plt #import matplotlip sebagai plt
fig, ax = plt.subplots() #fig dan ax menggunakan subplots untuk membuat gambar
ax.errorbar(depth_acc[:,0], depth_acc[:,1], yerr=depth_acc[:,2]) #membuat error bar kemudian grafik akan ditampilkan menggunakan show
plt.show() #menampilkan plot dari data yang ada