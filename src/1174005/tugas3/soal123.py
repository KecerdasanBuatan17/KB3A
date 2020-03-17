# In[44]: Soal1

import pandas as oni #melakukan import pada library pandas sebagai oni

laptop = {"Nama Laptop" : ['Asus','HP','Lenovo','Samsung']} #membuat varibel yang bernama laptop, dan mengisi dataframe nama2 laptop
x = oni.DataFrame(laptop) #variabel x membuat DataFrame dari library pandas dan akan memanggil variabel laptop. 
print (' Oni Punya Laptop ' + x) #print hasil dari x

# In[44]: Soal2

import numpy as oni
matrix_x = oni.eye(10) #membuat matrix dengan numpy dengan menggunakan fungsi eye
matrix_x #deklrasikan matrix_x yang telah dibuat

print (matrix_x) #print matrix_x yang telah dibuat dengan 10x10


# In[44]: Soal3

import matplotlib.pyplot as oni #import matploblib sebagai fahmi

oni.plot([1,1,7,4,0,0,5]) #memberikan nilai plot atau grafik pada fahmi
oni.xlabel('Oniwaldus Bere Mali') #memberikan label pada x
oni.ylabel('1174021') #memberikan label pada y
oni.show() #print hasil plot berbentuk grafik