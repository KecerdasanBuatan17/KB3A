# In[44]: Soal1

import pandas as habib #melakukan import pada library pandas sebagai habib

mobil = {"Nama Mobil" : ['Ferrari','Lamborgini',]} #membuat varibel yang bernama mobil
x = habib.DataFrame(mobil) #variabel x membuat DataFrame dari library pandas dan akan memanggil variabel mobil. 
print (' Habib Mempunyai Mobil ' + x) #print hasil dari x

# In[44]: Soal2

import numpy as habib #melakukan import numpy sebagai habib

matrix_x = habib.eye(16) #membuat matrix dengan numpy dengan menggunakan fungsi eye
matrix_x #deklrasikan matrix_x yang telah dibuat

print (matrix_x) #print matrix_x yang telah dibuat dengan 10x10


# In[44]: Soal3

import matplotlib.pyplot as habib #import matploblib sebagai habib

habib.plot([1,3,5,4,0,6,1]) #memberikan nilai plot atau grafik pada habib
habib.xlabel('Habib Abdul Rasyid') #memberikan label pada x
habib.ylabel('1174002') #memberikan label pada y
habib.show() #print hasil plot berbentuk grafik