# In[44]: Soal1

import pandas as felix #melakukan import pada library pandas sebagai felix

laptop = {"Nama Laptop" : ['Asus','HP','Lenovo','Samsung']} #membuat varibel yang bernama laptop, dan mengisi dataframe nama2 laptop
x = felix.DataFrame(laptop) #variabel x membuat DataFrame dari library pandas dan akan memanggil variabel laptop. 
print (' felix Punya Laptop ' + x) #print hasil dari x

# In[44]: Soal2

import numpy as felix #melakukan import numpy sebagai felix

matrix_x = felix.eye(10) #membuat matrix dengan numpy dengan menggunakan fungsi eye
matrix_x #deklrasikan matrix_x yang telah dibuat

print (matrix_x) #print matrix_x yang telah dibuat dengan 10x10


# In[44]: Soal3

import matplotlib.pyplot as felix #import matploblib sebagai felix

felix.plot([1,1,7,4,0,2,1]) #memberikan nilai plot atau grafik pada felix
felix.xlabel('Muhammad felix') #memberikan label pada x
felix.ylabel('1174026') #memberikan label pada y
felix.show() #print hasil plot berbentuk grafik