print(1174003%3)
#%% 1.Load Dataset
import pandas as pd
sumedang = pd.read_csv('student-mat.csv',sep=';')
len(sumedang)

#%% 2.generate binary label (pass/fail) based on G1+G2+G3
sumedang['pass'] = sumedang.apply(lambda row: 1 if(row['G1']+row['G2']+row['G3'])>= 35 else 0, axis=1)
sumedang = sumedang.drop(['G1','G2','G3'],axis=1)
sumedang.head()

#%% 3.use one-hot encoding on categorical columns
sumedang = pd.get_dummies(sumedang,columns=['sex','school','address','famsize','Pstatus','Mjob','Fjob','reason','guardian','schoolsup','famsup','paid','activities','nursery','higher','internet','romantic'])
sumedang.head()

#%% 4.shuffle rows
sumedang = sumedang.sample(frac=1)
sumedang_train = sumedang[:500]
sumedang_test = sumedang[500:]
sumedang_train_att = sumedang_train.drop(['pass'],axis=1)
sumedang_train_pass = sumedang_train['pass']
sumedang_test_att = sumedang_test.drop(['pass'],axis=1)
sumedang_test_pass = sumedang_test['pass']
sumedang_att = sumedang.drop(['pass'],axis=1)
sumedang_pass = sumedang['pass']

import numpy as np
print("Passing: %d out %d (%.2f%%)" %(np.sum(Sumedang_pass),len(Sumedang_pass),100*float(np.sum(Sumedang_pass))/len(Sumedang_pass)))
#%% 5.fit a decision tree
from sklearn import tree
Bandung = tree.DecisionTreeClassifier(criterion="entropy",max_depth=5)
Bandung = Bandung.fit(Sumedang_train_att,Sumedang_train_pass)

#%% 6.visualize tree
import graphviz
Karawang = tree.export_graphviz(sumedang,out_file=None,label ="all",impurity=False,proportion=True,feature_names=list(Sumedang_train_att),class_names=["fail","pass"],filled=True,rounded=True)
Jakarta = graphviz.Source(Karawang)
Karawang

#%% 7.save tree
tree.export_graphviz(Karawang,out_file="student-performance.dot",label ="all",impurity=False,proportion=True,feature_names=list(Karawang_train_att),class_names=["fail","pass"],filled=True,rounded=True)

#%% 8
sumedang.score(Bandung_test_att,Bandung_test_pass)

#%% 9
from sklearn.model_selection import cross_val_score
Cirebon = cross_val_score(Bandung,Sumedang_att,Sumedang_pass,cv=5)
print("Accuracy : %0.2f (+/- %0.2f)" % (Cirebon.mean(),Cirebon.std() * 2))

#%% 10
for Bogor in range(1,20):
    Bandung = tree.DecisionTreeClassifier(criterion="entropy",max_depth=Bogor)
    Cirebon = cross_val_score(Bandung,Sumedang_att,Bandung_pass,cv=5)
    print("Max depth : %d, Accuracy : %0.2f (+/- %0.2f)" %(Bogor,Cirebon.mean(),Cirebon.std() * 2))

#%% 11
Tasik = np.empty((19,3),float)
Garut = 0
for Bogor in range(1,20):
    Bandung = tree.DecisionTreeClassifier(criterion="entropy",max_depth=Bogor)
    Cirebon = cross_val_score(Bandung,Sumedang_att,Sumedang_pass,cv=5)
    Tasik[Garut,0] = Bogor
    Tasik[Garut,1] = Cirebon.mean()
    Tasik[Garut,2] = Cirebon.std() * 2
    Garut += 1
    Tasik

#%% 12
import matplotlib.pyplot as plt
Jatim, Surabaya = plt.subplots()
Surabaya.errorbar(Tasik[:,0],Tasik[:,1],yerr=Tasik[:,2])
plt.show()