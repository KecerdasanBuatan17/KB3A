from sklearn import datasets #mengimport datasets dari library sklearn
digits = datasets.load_digits() #meload datasets digits dan ditampung di variable digits
print(digits.data) #menampilkan data dari datasets digits
print(digits.target) #menampilkan target dari datasets digits