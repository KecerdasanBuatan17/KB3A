# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 19:15:16 2020

@author: IkrimaNingrrum
"""

print(116014%3)


#%% 1.Load Dataset
import pandas as pd
cianjur = pd.read_csv('student-mat.csv',sep=';')
len(cianjur)

#%% 2.generate binary label (pass/fail) based on G1+G2+G3
cianjur['pass'] = cianjur.apply(lambda row: 1 if(row['G1']+row['G2']+row['G3'])>= 35 else 0, axis=1)
cianjur = cianjur.drop(['G1','G2','G3'],axis=1)
cianjur.head()

#%% 3.use one-hot encoding on categorical columns
cianjur = pd.get_dummies(cianjur,columns=['sex','school','address','famsize','Pstatus','Mjob','Fjob','reason','guardian','schoolsup','famsup','paid','activities','nursery','higher','internet','romantic'])
cianjur.head()

#%% 4.shuffle rows
cianjur = cianjur.sample(frac=1)
cianjur_train = cianjur[:500]
cianjur_test = cianjur[500:]
cianjur_train_att = cianjur_train.drop(['pass'],axis=1)
cianjur_train_pass = cianjur_train['pass']
cianjur_test_att = cianjur_test.drop(['pass'],axis=1)
cianjur_test_pass = cianjur_test['pass']
cianjur_att = cianjur.drop(['pass'],axis=1)
cianjur_pass = cianjur['pass']

import numpy as np
print("Passing: %d out %d (%.2f%%)" %(np.sum(cianjur_pass),len(cianjur_pass),100*float(np.sum(cianjur_pass))/len(cianjur_pass)))
#%% 5.fit a decision tree
from sklearn import tree
bogor = tree.DecisionTreeClassifier(criterion="entropy",max_depth=5)
bogor = bogor.fit(cianjur_train_att,cianjur_train_pass)

#%% 6.visualize tree
import graphviz
yogyakarta = tree.export_graphviz(bogor,out_file=None,label ="all",impurity=False,proportion=True,feature_names=list(cianjur_train_att),class_names=["fail","pass"],filled=True,rounded=True)
malang = graphviz.Source(yogyakarta)
malang

#%% 7.save tree
tree.export_graphviz(yogyakarta,out_file="student-performance.dot",label ="all",impurity=False,proportion=True,feature_names=list(cianjur_train_att),class_names=["fail","pass"],filled=True,rounded=True)

#%% 8
bogor.score(cianjur_test_att,cianjur_test_pass)

#%% 9
from sklearn.model_selection import cross_val_score
makasar = cross_val_score(bogor,cianjur_att,cianjur_pass,cv=5)
print("Accuracy : %0.2f (+/- %0.2f)" % (makasar.mean(),makasar.std() * 2))

#%% 10
for sintang in range(1,20):
    bogor = tree.DecisionTreeClassifier(criterion="entropy",max_depth=sintang)
    makasar = cross_val_score(bogor,cianjur_att,cianjur_pass,cv=5)
    print("Max depth : %d, Accuracy : %0.2f (+/- %0.2f)" %(sintang,makasar.mean(),makasar.std() * 2))

#%% 11
pontianak = np.empty((19,3),float)
sidoarjo = 0
for sintang in range(1,20):
    bogor = tree.DecisionTreeClassifier(criterion="entropy",max_depth=sintang)
    makasar = cross_val_score(bandung,subang_att,subang_pass,cv=5)
    pontianak[garut,0] = sintang
    pontianak[garut,1] = makasar.mean()
    pontianak[garut,2] = makasar.std() * 2
    garut += 1
    pontianak

#%% 12
import matplotlib.pyplot as plt
sukabumi, tasik = plt.subplots()
tasik.errorbar(pontianak[:,0],pontianak[:,1],yerr=pontianak[:,2])
plt.show()