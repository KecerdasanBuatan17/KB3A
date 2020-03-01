# -*- coding: utf-8 -*-
"""
Created on Sun Mar  1 17:56:15 2020

@author: Nico Sembiring
"""

#%% 1 Loading an example dataset

from sklearn import datasets # Digunakan Untuk Memanggil class datasets dari library sklearn
iris = datasets.load_iris() # Menggunakan contoh datasets iris
digits = datasets.load_digits() # Menyimpan nilai data sets iris pada variabel digits

print(digits.target) #menampilkan hasil dari variabel digits

#%% 2 Learning and predicting

from sklearn import svm #digunakan untuk memanggil svm di library sklearn
clf = svm.SVC(gamma=0.001, C=100.)   #memberikan  nilai gamma secara manual 
clf.fit(digits.data[:-1], digits.target[:-1]) #clf sebagai classifier dan kemudian set latihan dengan metode fit
clf.predict(digits.data[-1:]) #memprediksi nilai baru dar digits.data

#%% 3 Model persistence

from sklearn import svm #digunakan untuk memanggil svm di library sklearn
from sklearn import datasets # Diguankan untuk class datasets dari library sklearn
clf = svm.SVC()  # membuat variabel clf, dan memanggil class svm dan fungsi SVC
X, y = datasets.load_iris(return_X_y=True) #Mengambil dataset iris dan mengembalikan nilainya.
clf.fit(X, y)  #Perhitungan nilai label

#%%
import pickle #memanggil library pickle 
s = pickle.dumps(clf) #untuk membuat variavel s sebagai classifier
clf2 = pickle.loads(s) # variabel clf2 sebagai load(s)
clf2.predict(X[0:1]) # untuk prediksi
#%%
from joblib import dump, load #memanggil dump, load melalui library joblib
dump(clf, '1174021.joblib')   #Menyimpan model kedalam 1174021.joblib
clf = load('1174021.joblib')   #Memanggil model 1174021

print(clf) #menampilkan hasil model clf

#%% 4 Conventions
import numpy as np # memanggil library numpy dan dibuat alias np
from sklearn import random_projection #Memanggil class random_projection pada library sklean

rng = np.random.RandomState(0) #Membuat variabel rng, dan mendefisikan np, fungsi random dan attr RandomState kedalam variabel
X = rng.rand(10, 2000) # untuk membuat variabel X, dan menentukan nilai random dari 10 - 2000
X = np.array(X, dtype='float32') # untuk menyimpan hasil nilai random sebelumnya, kedalam array, dan menentukan typedatanya sebagai float32
X.dtype  # Mengubah data tipe menjadi float64
 
transformer = random_projection.GaussianRandomProjection() #membuat variabel transformer, dan mendefinisikan classrandom_projection dan memanggil fungsi GaussianRandomProjection
X_new = transformer.fit_transform(X) # membuat variabel baru dan melakukan perhitungan label pada variabel X
X_new.dtype # Mengubah data tipe menjadi float64

print(X_new) #menampilkan hasil