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
dump(clf, '1174006.joblib') #memanggil method dumps dengan argumen clf dan nama file joblibnya
clf3 = load('1174006.joblib')#memanggil method loads dengan argumen nama file joblibnya dan ditampung di variable clf3
print(clf3.predict(X[0:1])) #menampilkan hasil dari method predict dengan argumen data variable X pertama