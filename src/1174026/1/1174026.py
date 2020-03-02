# -*- coding: utf-8 -*-
"""
Created on Sun Mar  1 17:56:15 2020

@author: Felix Lase
"""

#%% 1 Loading an example dataset

from sklearn import datasets # Digunakan Untuk Memanggil class datasets dari library sklearn
iris = datasets.load_iris()  # Menggunakan contoh datasets iris
x = iris.data              # Menyimpan nilai data sets iris pada variabel x
y = iris.target            # Menyimpan nilai data label iris pada variabel y    

#%% 2 Learning and predicting

from sklearn.neighbors import KNeighborsClassifier #Digunakan Untuk Memanggil fungsi KNeighborsClassifier
                                                  # pada class sklearn dan library sklearn
import numpy as np # memanggil library numpy dan dibuat alias np
knn=KNeighborsClassifier(n_neighbors=1) #membuat variabel kkn, dan memanggil fungsi KNeighborsClassifier
                                        #dan mendefinisikan k-nya adalah 1
knn.fit(x,y)                            #Perhitungan matematika library kkn
a=np.array([1.0,2.0,3.0,4.0])           #Membuat Array
a = a.reshape(1,-1)                     #Mengubah Bentuk Array jadi 1 dimensi
hasil = knn.predict(a)                  #Memanggil fungsi prediksi
print(hasil)   

#%% 3 Model persistence

from sklearn import svm  # Digunakan untuk memangil class svm dari library sklearn
from sklearn import datasets # Diguankan untuk class datasets dari library sklearn
clf = svm.SVC()              # membuat variabel clf, dan memanggil class svm dan fungsi SVC
X, y = datasets.load_iris(return_X_y=True) #Mengambil dataset iris dan mengembalikan nilainya.
clf.fit(X, y)               #Perhitungan nilai label

from joblib import dump, load #memanggil class dump dan load pada library joblib
dump(clf, '1174027.joblib') #Menyimpan model kedalam 1174026.joblib
hasil = load('1174027.joblib') #Memanggil model 1174026
print(hasil) # Menampilkan Model yang dipanggil sebelumnya

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