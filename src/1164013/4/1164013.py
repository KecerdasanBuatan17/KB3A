# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 09:27:45 2020

@author: Ikrima
"""

# In[soal 1]:
#mengimport library pandas dan menamainya coco
import pandas as sayang
#membuat variable bernama harun dan mengisinya dengan data dari dataset mine yang telah dibuat
aku = sayang.read_csv("mine.csv") 
#untuk melihat 5 baris pertama dari data harun
atas = aku.head() 
#untuk mengetahui berapa banyak baris data
aku.shape 
#menampilkan isi dari varibale c pada console
print(atas)
# In[soal 2]:
#memasukkan 450 data pertama ke dalam variable datatraining
datatraining = saya[:450] 
#memasukkan 50 data terakhir kedalam variable datatesting
datatesting = saya[450:]
# In[soal 3]:
print(1164013%4) #menghitung NPM Mod 4 untuk menentukan data yang akan digunakan
#mengimport library pandas dan menamainya tampan
import pandas as tampan
data=tampan.read_csv("Youtube02-KatyPerry.csv")
spam=data.query('CLASS == 1')
nospam=data.query('CLASS == 0')
#%% melakukan fungsi bag of word dengan cara menghitung semua kata
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()
#%% melakukan bag of word pada dataframe pada colom CONTENT
data_vektorisasi = vectorizer.fit_transform(data['CONTENT'])
#%% melihat isi vektorisasi
data_vektorisasi
#%% menampilkan isi data pada baris ke 300
print(data['CONTENT'][300])
#%% untuk mengambil apa saja nama kolom yang tersedia
dk=vectorizer.get_feature_names()
#%%: randomisasi agar hasil sempurna pada saat klasifikasi
dshuf = data.sample(frac=1)
#%%: membuat data traning dan testing
dk_train=dshuf[:300]
dk_test=dshuf[300:]
#%%: melakukan training pada data training dan di vektorisasi
dk_train_att=vectorizer.fit_transform(dk_train['CONTENT'])
print(dk_train_att)
#%% melakukan testing pada data testing dan di vektorisasi
dk_test_att=vectorizer.transform(dk_test['CONTENT'])
print(dk_test_att)
#%%: Dimana akan mengambil label spam dan bukan spam
dk_train_label=dk_train['CLASS']
print(dk_train_label)
dk_test_label=dk_test['CLASS']
print(dk_test_label)
# In[soal 4]:
#klasifikasi SVM
from sklearn import svm #import librari svm dari sklearn
clfsvm = svm.SVC()#membuat variabel clfsvm berisikan method svc
#variabel tersebut di berikan method fit dengan isian data train vektorisasi dan data training label
clfsvm.fit(dk_train_att, dk_train_label)
clfsvm.score(dk_test_att, dk_test_label)
# In[soal 5]:
#klasifikasi decision tree
from sklearn import tree #import library tree dari sklearn
clftree = tree.DecisionTreeClassifier()
clftree.fit(dk_train_att, dk_train_label)
clftree.score(dk_test_att, dk_test_label)
# In[soal 6]:
##plot comfusion matrix
from sklearn.metrics import confusion_matrix
pred_labels = clftree.predict(dk_test_att)
cm = confusion_matrix(dk_test_label, pred_labels)
cm
# In[soal 7]:
#cross valodation
from sklearn.model_selection import cross_val_score
scores = cross_val_score(clftree,dk_train_att,dk_train_label,cv=5)
scorerata2=scores.mean()
scorersd=scores.std()
#%%:
from sklearn.model_selection import cross_val_score
scores = cross_val_score(clftree, dk_train_att, dk_train_label, cv=5)
# show average score and +/- two standard deviations away (covering 95
#% of scores)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(),
scores.std() * 2))
#%%:
scorestree = cross_val_score(clftree, dk_train_att, dk_train_label, cv=5)
print("Accuracy: %0.2f (+/- %0.2f)" % (scorestree.mean(),
scorestree.std() * 2))
#%%:
scoressvm = cross_val_score(clfsvm, dk_train_att, dk_train_label, cv=5)
print("Accuracy: %0.2f (+/- %0.2f)" % (scoressvm.mean(),
scoressvm.std() * 2))
# In[soal 8]:
#Pengamatan program
import numpy as np
from sklearn.ensemble import RandomForestClassifier
max_features_opts = range(1, 10, 1)
n_estimators_opts = range(2, 40, 4)
rf_params = np.empty((len(max_features_opts)*len(n_estimators_opts),4) , float)
i = 0
for max_features in max_features_opts:
    for n_estimators in n_estimators_opts:
        clf = RandomForestClassifier(max_features=max_features, n_estimators=n_estimators)
        scores = cross_val_score(clf, dk_train_att, dk_train_label, cv=5)
        rf_params[i,0] = max_features
        rf_params[i,1] = n_estimators
        rf_params[i,2] = scores.mean()
        rf_params[i,3] = scores.std() * 2
        i += 1
        print("Max features: %d, num estimators: %d, accuracy: %0.2f (+/- %0.2f)"
% (max_features, n_estimators, scores.mean(), scores.std() * 2))

#%%
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
fig = plt.figure()
fig.clf()
ax = fig.gca(projection='3d')
x = rf_params[:,0]
y = rf_params[:,1]
z = rf_params[:,2]
ax.scatter(x, y, z)
ax.set_zlim(0.6, 1)
ax.set_xlabel('Max features')
ax.set_ylabel('Num estimators')
ax.set_zlabel('Avg accuracy')
plt.show()