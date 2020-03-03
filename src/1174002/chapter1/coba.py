# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 00:32:46 2020

@author: Habib Abdul R
"""

from sklearn import datasets 
iris = datasets.load_iris() 
digits = datasets.load_digits()

#%% contoh 1
from sklearn import datasets #mengimport datasets dari library sklearn 
digits = datasets.load_digits() #meload datasets digits dan ditampung di variable digits 
print(digits.data) #menampilkan data dari datasets digits 
print(digits.target) #menampilkan target dari datasets digits
#%% contoh 2
from sklearn import svm, datasets #mengimport svm dan datasets dari library sklearn
digits = datasets.load_digits() #meload datasets digits dan ditampung di variable digits
clf = svm.SVC(gamma=0.001, C=100.) #memanggil class SVC (Support Vector Classification) dan menset argument constructor SVC serta ditampung di variable clf
clf.fit(digits.data[:-1], digits.target[:-1]) #memanggil method fit untuk melakukan training data dengan argumen data dan target dari datasets digits dimana data dan target terakhir tidak dipakai
print(clf.predict(digits.data[-1:])) #menampilkan hasil dari method predict dengan argumen data digits terkakhir
#%% contoh 3
from sklearn import svm, datasets #mengimport svm dan datasets dari library sklearn
clf = svm.SVC(gamma=0.001, C=100.) #memanggil class SVC (Support Vector Classification) dan menset argument constructor SVC serta ditampung di variable clf
X, y = datasets.load_iris(return_X_y=True) #meload datasets iris dan ditampung di variable x untuk data dan y untuk target
clf.fit(X, y) #memanggil method fit untuk melakukan training data dengan argumen data dan target dari datasets iris

#Pickle
import pickle #mengimport pickle
s = pickle.dumps(clf) #memanggil method dumps dengan argumen clf dan ditampung di variable s
clf2 = pickle.loads(s) #memanggil method loads dengan argumen s dan ditampung di variable clf2
print(clf2.predict(X[0:1])) #menampilkan hasil dari method predict dengan argumen data variable X pertama

#Joblib
from joblib import dump, load #mengimport dump dan load dari library joblib
dump(clf, '1174002.joblib') #memanggil method dumps dengan argumen clf dan nama file joblibnya
clf3 = load('1174002.joblib')#memanggil method loads dengan argumen nama file joblibnya dan ditampung di variable clf3
print(clf3.predict(X[0:1])) #menampilkan hasil dari method predict dengan argumen data variable X pertama

#%% contoh 4
#Type casting
import numpy as np #mengimport library numpy
from sklearn import random_projection #mengimport library random_projection
rng = np.random.RandomState(0) #membuat random number
X = rng.rand(10, 2000) #membuat array 2D dengan random number
X = np.array(X, dtype='float32') #membuat array dengan argumen list datanya dan type datanya
print(X.dtype) #menampilak type datanya
transformer = random_projection.GaussianRandomProjection() #mengurangi dimensionality
X_new = transformer.fit_transform(X) #melatih data, lalu di tranformasikan ke dataset secara khusus
print(X_new.dtype) #menampilkan type datanya

from sklearn import datasets #mengimport library datasets
from sklearn.svm import SVC #mengimport library SVC
iris = datasets.load_iris()  #meload datasets iris dan ditampung di variable iris
clf = SVC(gamma=0.001, C=100.) #memanggil class SVC (Support Vector Classification) dan menset argument constructor SVC serta ditampung di variable clf
clf.fit(iris.data, iris.target) #melatih data degan argumen data dan ntarget
print(list(clf.predict(iris.data[:3]))) #menampilkan hasil prediksinya
clf.fit(iris.data, iris.target_names[iris.target]) #melatih data degan argumen data dan nama dari target
print(list(clf.predict(iris.data[:3]))) #menampilkan hasil prediksinya

#Refitting and updating parameters
import numpy as np  #mengimport library numpy
from sklearn.datasets import load_iris #mengimport dataset iris 
from sklearn.svm import SVC  #mengimport library SVC
X, y = load_iris(return_X_y=True)  #meload datasets iris dan ditampung di variable x untuk data dan y untuk target
clf = SVC(gamma=0.001, C=100.) #memanggil class SVC (Support Vector Classification) dan menset argument constructor SVC serta ditampung di variable clf
clf.set_params(kernel='linear').fit(X, y) #meset paramsnya dimana argumennya kernel yang dipakai linear, lalu melatih data
print(clf.predict(X[:5])) #menampilkan hasil prediksinya
clf.set_params(kernel='rbf').fit(X, y) #meset paramsnya dimana argumennya kernel yang dipakai rbf, lalu melatih data
print(clf.predict(X[:5])) #menampilkan hasil prediksinya

#Multiclass vs. Multilabel Fitting
from sklearn.svm import SVC  #mengimport library SVC
from sklearn.multiclass import OneVsRestClassifier #mengimport library OneVsRestClassifier
from sklearn.preprocessing import LabelBinarizer #mengimport library LabelBinarizer
X = [[1, 2], [2, 4], [4, 5], [3, 2], [3, 1]] #membuat list
y = [0, 0, 1, 1, 2] #membuat list
classif = OneVsRestClassifier(estimator=SVC(random_state=0, gamma=0.001, C=100.)) #multiclass or multilabel classification estimator yang digunakan
print(classif.fit(X, y).predict(X)) #menampilkan hasil prediksinya
y = LabelBinarizer().fit_transform(y) #melabelkan data
print(classif.fit(X, y).predict(X)) #menampilkan hasil prediksinya

from sklearn.preprocessing import MultiLabelBinarizer #mengimport library MultiLabelBinarizer
y = [[0, 1], [0, 2], [1, 3], [0, 2, 3], [2, 4]] #membuat list
y = MultiLabelBinarizer().fit_transform(y) #multilable data,kemudian melatih data, lalu di tranformasikan ke dataset secara khusus
print(classif.fit(X, y).predict(X)) #menampilkan hasil prediksinya