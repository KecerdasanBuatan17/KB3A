# -*- coding: utf-8 -*-
"""
Created on Sat Mar  7 12:03:42 2020

@author: ACER
"""

print(1174031%3)
#%% 1.Load Dataset
import pandas as pd
apel = pd.read_csv('student-mat.csv',sep=';')
len(apel)

#%% 2.generate binary label (pass/fail) based on G1+G2+G3
apel['pass'] = apel.apply(lambda row: 1 if(row['G1']+row['G2']+row['G3'])>= 35 else 0, axis=1)
apel = apel.drop(['G1','G2','G3'],axis=1)
apel.head()

#%% 3.use one-hot encoding on categorical columns
apel = pd.get_dummies(apel,columns=['sex','school','address','famsize','Pstatus','Mjob','Fjob','reason','guardian','schoolsup','famsup','paid','activities','nursery','higher','internet','romantic'])
apel.head()

#%% 4.shuffle rows
apel = apel.sample(frac=1)
apel_train = apel[:500]
apel_test = apel[500:]
apel_train_att = apel_train.drop(['pass'],axis=1)
apel_train_pass = apel_train['pass']
apel_test_att = apel_test.drop(['pass'],axis=1)
apel_test_pass = apel_test['pass']
apel_att = apel.drop(['pass'],axis=1)
apel_pass = apel['pass']

import numpy as np
print("Passing: %d out %d (%.2f%%)" %(np.sum(apel_pass),len(apel_pass),100*float(np.sum(apel_pass))/len(apel_pass)))
#%% 5.fit a decision tree
from sklearn import tree
anggur = tree.DecisionTreeClassifier(criterion="entropy",max_depth=5)
anggur = anggur.fit(apel_train_att,apel_train_pass)

#%% 6.visualize tree
import graphviz
nanas = tree.export_graphviz(anggur,out_file=None,label ="all",impurity=False,proportion=True,feature_names=list(apel_train_att),class_names=["fail","pass"],filled=True,rounded=True)
pear = graphviz.Source(nanas)
pear

#%% 7.save tree
tree.export_graphviz(anggur,out_file="student-performance.dot",label ="all",impurity=False,proportion=True,feature_names=list(apel_train_att),class_names=["fail","pass"],filled=True,rounded=True)

#%% 8
anggur.score(apel_test_att,apel_test_pass)

#%% 9
from sklearn.model_selection import cross_val_score
pisang = cross_val_score(anggur,apel_att,apel_pass,cv=5)
print("Accuracy : %0.2f (+/- %0.2f)" % (pisang.mean(),pisang.std() * 2))

#%% 10
for tomat in range(1,20):
    anggur = tree.DecisionTreeClassifier(criterion="entropy",max_depth=tomat)
    pisang = cross_val_score(anggur,apel_att,apel_pass,cv=5)
    print("Max depth : %d, Accuracy : %0.2f (+/- %0.2f)" %(tomat,pisang.mean(),pisang.std() * 2))

#%% 11
durian = np.empty((19,3),float)
jeruk = 0
for tomat in range(1,20):
    anggur = tree.DecisionTreeClassifier(criterion="entropy",max_depth=jeruk)
    pisang = cross_val_score(anggur,apel_att,apel_pass,cv=5)
    durian[jeruk,0] = tomat
    durian[jeruk,1] = pisang.mean()
    durian[jeruk,2] = pisang.std() * 2
    jeruk += 1
    durian

#%% 12
import matplotlib.pyplot as plt
salak, pepaya = plt.subplots()
pepaya.errorbar(durian[:,0],durian[:,1],yerr=durian[:,2])
plt.show()