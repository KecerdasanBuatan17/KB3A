# In[44]: Soal1

import pandas as fahmi #melakukan import pada library pandas sebagai fahmi

laptop = {"Nama Laptop" : ['Asus','HP','Lenovo','Samsung']} #membuat varibel yang bernama laptop, dan mengisi dataframe nama2 laptop
x = fahmi.DataFrame(laptop) #variabel x membuat DataFrame dari library pandas dan akan memanggil variabel laptop. 
print (' Fahmi Punya Laptop ' + x) #print hasil dari x

# In[44]: Soal2

import numpy as fahmi #melakukan import numpy sebagai fahmi

matrix_x = fahmi.eye(10) #membuat matrix dengan numpy dengan menggunakan fungsi eye
matrix_x #deklrasikan matrix_x yang telah dibuat

print (matrix_x) #print matrix_x yang telah dibuat dengan 10x10


# In[44]: Soal3

import matplotlib.pyplot as fahmi #import matploblib sebagai fahmi

fahmi.plot([1,1,7,4,0,2,1]) #memberikan nilai plot atau grafik pada fahmi
fahmi.xlabel('Muhammad Fahmi') #memberikan label pada x
fahmi.ylabel('1174021') #memberikan label pada y
fahmi.show() #print hasil plot berbentuk grafik