from sklearn import svm, datasets #mengimport svm dan datasets dari library sklearn
digits = datasets.load_digits() #meload datasets digits dan ditampung di variable digits
clf = svm.SVC(gamma=0.001, C=100.) #memanggil class SVC (Support Vector Classification) dan menset argument constructor SVC serta ditampung di variable clf
clf.fit(digits.data[:-1], digits.target[:-1]) #memanggil method fit untuk melakukan training data dengan argumen data dan target dari datasets digits dimana data dan target terakhir tidak dipakai
print(clf.predict(digits.data[-1:])) #menampilkan hasil dari method predict dengan argumen data digits terkakhir