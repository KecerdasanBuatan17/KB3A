# -*- coding: utf-8 -*-
"""
Created on Sun Mar  8 19:27:22 2020

@author: Sujadi
"""

print(1174051%3)
#%% 1.Load Dataset
import pandas as pd
kediri = pd.read_csv('student-mat.csv',sep=';')
len(kediri)

#%% 2.generate binary label (pass/fail) based on G1+G2+G3
kediri['pass'] = kediri.apply(lambda row: 1 if(row['G1']+row['G2']+row['G3'])>= 35 else 0, axis=1)
kediri = kediri.drop(['G1','G2','G3'],axis=1)
kediri.head()

#%% 3.use one-hot encoding on categorical columns
kediri = pd.get_dummies(kediri,columns=['sex','school','address','famsize','Pstatus','Mjob','Fjob','reason','guardian','schoolsup','famsup','paid','activities','nursery','higher','internet','romantic'])
kediri.head()

#%% 4.shuffle rows
kediri = kediri.sample(frac=1)
kediri_train = kediri[:500]
kediri_test = kediri[500:]
kediri_train_att = kediri_train.drop(['pass'],axis=1)
kediri_train_pass = kediri_train['pass']
kediri_test_att = kediri_test.drop(['pass'],axis=1)
kediri_test_pass = kediri_test['pass']
kediri_att = kediri.drop(['pass'],axis=1)
kediri_pass = kediri['pass']

import numpy as np
print("Passing: %d out %d (%.2f%%)" %(np.sum(kediri_pass),len(kediri_pass),100*float(np.sum(kediri_pass))/len(kediri_pass)))
#%% 5.fit a decision tree
from sklearn import tree
malang = tree.DecisionTreeClassifier(criterion="entropy",max_depth=5)
malang = malang.fit(kediri_train_att,kediri_train_pass)

#%% 6.visualize tree
import graphviz
madiun = tree.export_graphviz(malang,out_file=None,label ="all",impurity=False,proportion=True,feature_names=list(kediri_train_att),class_names=["fail","pass"],filled=True,rounded=True)
magetan = graphviz.Source(madiun)
magetan

#%% 7.save tree
tree.export_graphviz(malang,out_file="student-performance.dot",label ="all",impurity=False,proportion=True,feature_names=list(kediri_train_att),class_names=["fail","pass"],filled=True,rounded=True)

#%% 8
malang.score(kediri_test_att,kediri_test_pass)

#%% 9
from sklearn.model_selection import cross_val_score
batu = cross_val_score(malang,kediri_att,kediri_pass,cv=5)
print("Accuracy : %0.2f (+/- %0.2f)" % (batu.mean(),batu.std() * 2))

#%% 10
for surabaya in range(1,20):
    malang = tree.DecisionTreeClassifier(criterion="entropy",max_depth=surabaya)
    batu = cross_val_score(malang,kediri_att,kediri_pass,cv=5)
    print("Max depth : %d, Accuracy : %0.2f (+/- %0.2f)" %(surabaya,batu.mean(),batu.std() * 2))

#%% 11
jember = np.empty((19,3),float)
probolinggo = 0
for surabaya in range(1,20):
    malang = tree.DecisionTreeClassifier(criterion="entropy",max_depth=surabaya)
    batu = cross_val_score(malang,kediri_att,kediri_pass,cv=5)
    jember[probolinggo,0] = surabaya
    jember[probolinggo,1] = batu.mean()
    jember[probolinggo,2] = batu.std() * 2
    probolinggo += 1
    jember

#%% 12
import matplotlib.pyplot as plt
situbondo, bondowoso = plt.subplots()
bondowoso.errorbar(jember[:,0],jember[:,1],yerr=jember[:,2])
plt.show()