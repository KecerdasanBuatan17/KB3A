# -*- coding: utf-8 -*-
"""
Created on Sun Mar  8 15:02:52 2020

@author: ROG
"""

print(1174026%3)
#%% 1.Load Dataset
import pandas as pd
nasigoreng = pd.read_csv('student-mat.csv',sep=';')
len(nasigoreng)

#%% 2.generate binary label (pass/fail) based on G1+G2+G3
nasigoreng['pass'] = nasigoreng.apply(lambda row: 1 if(row['G1']+row['G2']+row['G3'])>= 35 else 0, axis=1)
nasigoreng = nasigoreng.drop(['G1','G2','G3'],axis=1)
nasigoreng.head()

#%% 3.use one-hot encoding on categorical columns
nasigoreng = pd.get_dummies(nasigoreng,columns=['sex','school','address','famsize','Pstatus','Mjob','Fjob','reason','guardian','schoolsup','famsup','paid','activities','nursery','higher','internet','romantic'])
nasigoreng.head()

#%% 4.shuffle rows
nasigoreng = nasigoreng.sample(frac=1)
nasigoreng_train = nasigoreng[:500]
nasigoreng_test = nasigoreng[500:]
nasigoreng_train_att = nasigoreng_train.drop(['pass'],axis=1)
nasigoreng_train_pass = nasigoreng_train['pass']
nasigoreng_test_att = nasigoreng_test.drop(['pass'],axis=1)
nasigoreng_test_pass = nasigoreng_test['pass']
nasigoreng_att = nasigoreng.drop(['pass'],axis=1)
nasigoreng_pass = nasigoreng['pass']

import numpy as np
print("Passing: %d out %d (%.2f%%)" %(np.sum(nasigoreng_pass),len(nasigoreng_pass),100*float(np.sum(nasigoreng_pass))/len(nasigoreng_pass)))
#%% 5.fit a decision tree
from sklearn import tree
satebebek = tree.DecisionTreeClassifier(criterion="entropy",max_depth=5)
satebebek = satebebek.fit(nasigoreng_train_att,nasigoreng_train_pass)

#%% 6.visualize tree
import graphviz
pizza = tree.export_graphviz(satebebek,out_file=None,label ="all",impurity=False,proportion=True,feature_names=list(nasigoreng_train_att),class_names=["fail","pass"],filled=True,rounded=True)
sateayam = graphviz.Source(pizza)
sateayam

#%% 7.save tree
tree.export_graphviz(satebebek,out_file="student-performance.dot",label ="all",impurity=False,proportion=True,feature_names=list(nasigoreng_train_att),class_names=["fail","pass"],filled=True,rounded=True)

#%% 8
satebebek.score(nasigoreng_test_att,nasigoreng_test_pass)

#%% 9
from sklearn.model_selection import cross_val_score
mieayam = cross_val_score(satebebek,nasigoreng_att,nasigoreng_pass,cv=5)
print("Accuracy : %0.2f (+/- %0.2f)" % (mieayam.mean(),mieayam.std() * 2))

#%% 10
for baso in range(1,20):
    satebebek = tree.DecisionTreeClassifier(criterion="entropy",max_depth=baso)
    mieayam = cross_val_score(satebebek,nasigoreng_att,nasigoreng_pass,cv=5)
    print("Max depth : %d, Accuracy : %0.2f (+/- %0.2f)" %(baso,mieayam.mean(),mieayam.std() * 2))

#%% 11
satekambing = np.empty((19,3),float)
ayamgeprek = 0
for baso in range(1,20):
    satebebek = tree.DecisionTreeClassifier(criterion="entropy",max_depth=baso)
    mieayam = cross_val_score(satebebek,nasigoreng_att,nasigoreng_pass,cv=5)
    satekambing[ayamgeprek,0] = baso
    satekambing[ayamgeprek,1] = mieayam.mean()
    satekambing[ayamgeprek,2] = mieayam.std() * 2
    ayamgeprek += 1
    satekambing

#%% 12
import matplotlib.pyplot as plt
nasitelor, nasiayam = plt.subplots()
nasiayam.errorbar(satekambing[:,0],satekambing[:,1],yerr=satekambing[:,2])
plt.show()