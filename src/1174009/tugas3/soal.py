# In[1]

import pandas as yuli #melakukan import pada library pandas sebagai yuli

sayur = {"Nama" : ['Kangkung','Bayam','Sawi','Pokcoy']} #membuat varibel yang bernama sayur, dan mengisi dataframe nama2 sayur
x = yuli.DataFrame(sayur) #variabel x membuat DataFrame dari library pandas dan akan memanggil variabel sayur. 
print (' Aku akan masak sayur ' + x) #print hasil dari x

# In[44]: Soal2

import numpy as yuli #melakukan import numpy sebagai yuli

matrix_x = yuli.eye(15) #membuat matrix dengan numpy dengan menggunakan fungsi eye
matrix_x #deklrasikan matrix_x yang telah dibuat

print (matrix_x) #print matrix_x yang telah dibuat dengan 10x10


# In[44]: Soal3

import matplotlib.pyplot as yuli #import matploblib sebagai yuli

yuli.plot([0,2,0,5,8,1,5,7]) #memberikan nilai plot atau grafik pada yuli
yuli.xlabel('Yuli') #memberikan label pada x
yuli.ylabel('1174009') #memberikan label pada y
yuli.show() #print hasil plot berbentuk grafik