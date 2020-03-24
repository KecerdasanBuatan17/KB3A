# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 12:38:47 2020

@author: Arjun
"""

npm = 1174008%4
print(npm)
#%%
#Import pandas dan meload file Youtube05-Shakira
import pandas as pan
data_komentar = pan.read_csv('Youtube05-Shakira.csv')

#Mengelompokkan spam dan bukan spam ke 2 variabel yang berbeda
spam=data_komentar.query('CLASS == 1')
no_spam=data_komentar.query('CLASS == 0')

#Melakukan fungsi bag of word dengan cara menghitung semua kata yang ada pada file
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()

#Membuat variabel dengan perintah untuk memproses kolom CONTENT pada dataset, lalu membaca variable yang ada
data_vektor = vectorizer.fit_transform(data_komentar['CONTENT'])
data_vektor

#Menampilkan sampel komentar spam yang didapat
print(data_komentar['CONTENT'][304])

#Menampilkan daftar nama kolom
dk=vectorizer.get_feature_names()

#Untuk melakukan randomisasi dari setiap komentar
train_test = data_komentar.sample(frac=1)
#Memuat data training dan testing dari data yang sudah ada
dk_train=train_test[:300]
dk_test=train_test[300:]

#Melakukan training pada data training dan memvektorisasi data tersebut
dk_train_att = vectorizer.fit_transform(dk_train['CONTENT'])
dk_train_att

#Melakukan Testing pada data testing dan memvektorisasi data tersebut
dk_test_att=vectorizer.transform(dk_test['CONTENT'])
dk_test_att

#Mengambil label spam dan bukan spam
dk_train_label=dk_train['CLASS']
dk_test_label = dk_test['CLASS']

#%% 
#Coding untuk melakukan klasifikasi SVM
from sklearn import svm
clfsvm = svm.SVC()
clfsvm.fit(dk_train_att, dk_train_label)
clfsvm.score(dk_test_att, dk_test_label)


#%%
#Melakukan klasifikasi Decision Tree
from sklearn import tree
clftree = tree.DecisionTreeClassifier()
clftree.fit(dk_train_att, dk_train_label)
clftree.score(dk_test_att, dk_test_label)

#%%
#Melakukan confusion matrix
from sklearn.metrics import confusion_matrix
pred_labels = clftree.predict(dk_test_att)
cm = confusion_matrix(dk_test_label, pred_labels)
cm

#%%
from sklearn.model_selection import cross_val_score
scores = cross_val_score(clftree,dk_train_att,dk_train_label,cv=5)
scorerata2=scores.mean()
scorersd=scores.std()

from sklearn.model_selection import cross_val_score
scores = cross_val_score(clftree, dk_train_att, dk_train_label, cv=5)
# show average score and +/- two standard deviations away (covering 95
#% of scores)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

scorestree = cross_val_score(clftree, dk_train_att, dk_train_label, cv=5)
print("Accuracy: %0.2f (+/- %0.2f)" % (scorestree.mean(), scorestree.std() * 2))

scoressvm = cross_val_score(clfsvm, dk_train_att, dk_train_label, cv=5)
print("Accuracy: %0.2f (+/- %0.2f)" % (scoressvm.mean(), scoressvm.std() * 2))

#%%
from sklearn.ensemble import RandomForestClassifier
import numpy as np
max_features_opts = range(1, 10, 1)
n_estimators_opts = range(2, 40, 4)
rf_params = np.empty((len(max_features_opts)*len(n_estimators_opts),4), float)
i = 0
for max_features in max_features_opts:
    for n_estimators in n_estimators_opts:
        clf = RandomForestClassifier(max_features=max_features,n_estimators=n_estimators)
        scores = cross_val_score(clf, dk_train_att, dk_train_label, cv=5)
        rf_params[i,0] = max_features
        rf_params[i,1] = n_estimators
        rf_params[i,2] = scores.mean()
        rf_params[i,3] = scores.std() * 2
        i += 1
        print("Max features: %d, num estimators: %d, accuracy: %0.2f (+/- %0.2f)" 
%   (max_features, n_estimators, scores.mean(), scores.std() * 2))
        
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