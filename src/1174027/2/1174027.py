# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 18:00:31 2020

@author: Harun
"""

print(1174027%3)
#%% 1.Load Dataset
import pandas as pd
subang = pd.read_csv('student-mat.csv',sep=';')
len(subang)

#%% 2.generate binary label (pass/fail) based on G1+G2+G3
subang['pass'] = subang.apply(lambda row: 1 if(row['G1']+row['G2']+row['G3'])>= 35 else 0, axis=1)
subang = subang.drop(['G1','G2','G3'],axis=1)
subang.head()

#%% 3.use one-hot encoding on categorical columns
subang = pd.get_dummies(subang,columns=['sex','school','address','famsize','Pstatus','Mjob','Fjob','reason','guardian','schoolsup','famsup','paid','activities','nursery','higher','internet','romantic'])
subang.head()

#%% 4.shuffle rows
subang = subang.sample(frac=1)
subang_train = subang[:500]
subang_test = subang[500:]
subang_train_att = subang_train.drop(['pass'],axis=1)
subang_train_pass = subang_train['pass']
subang_test_att = subang_test.drop(['pass'],axis=1)
subang_test_pass = subang_test['pass']
subang_att = subang.drop(['pass'],axis=1)
subang_pass = subang['pass']

import numpy as np
print("Passing: %d out %d (%.2f%%)" %(np.sum(subang_pass),len(subang_pass),100*float(np.sum(subang_pass))/len(subang_pass)))
#%% 5.fit a decision tree
from sklearn import tree
bandung = tree.DecisionTreeClassifier(criterion="entropy",max_depth=5)
bandung = bandung.fit(subang_train_att,subang_train_pass)

#%% 6.visualize tree
import graphviz
bogor = tree.export_graphviz(bandung,out_file=None,label ="all",impurity=False,proportion=True,feature_names=list(subang_train_att),class_names=["fail","pass"],filled=True,rounded=True)
jakarta = graphviz.Source(bogor)
jakarta

#%% 7.save tree
tree.export_graphviz(bandung,out_file="student-performance.dot",label ="all",impurity=False,proportion=True,feature_names=list(subang_train_att),class_names=["fail","pass"],filled=True,rounded=True)

#%% 8
bandung.score(subang_test_att,subang_test_pass)

#%% 9
from sklearn.model_selection import cross_val_score
depok = cross_val_score(bandung,subang_att,subang_pass,cv=5)
print("Accuracy : %0.2f (+/- %0.2f)" % (depok.mean(),depok.std() * 2))

#%% 10
for surabaya in range(1,20):
    bandung = tree.DecisionTreeClassifier(criterion="entropy",max_depth=surabaya)
    depok = cross_val_score(bandung,subang_att,subang_pass,cv=5)
    print("Max depth : %d, Accuracy : %0.2f (+/- %0.2f)" %(surabaya,depok.mean(),depok.std() * 2))

#%% 11
medan = np.empty((19,3),float)
sidoarjo = 0
for surabaya in range(1,20):
    bandung = tree.DecisionTreeClassifier(criterion="entropy",max_depth=surabaya)
    depok = cross_val_score(bandung,subang_att,subang_pass,cv=5)
    medan[sidoarjo,0] = surabaya
    medan[sidoarjo,1] = depok.mean()
    medan[sidoarjo,2] = depok.std() * 2
    sidoarjo += 1
    medan

#%% 12
import matplotlib.pyplot as plt
blitar, kediri = plt.subplots()
kediri.errorbar(medan[:,0],medan[:,1],yerr=medan[:,2])
plt.show()