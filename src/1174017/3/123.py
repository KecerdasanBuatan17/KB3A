# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 15:28:34 2020

@author: Rifky
"""

# In[44]: Soal1

import pandas as Rifky #melakukan import pada library pandas sebagai Rifky

iphone = {"Nama Iphone" : ['Iphone7','Iphone8','Iphone10','Iphone11']} #membuat varibel yang bernama iphone, dan mengisi dataframe nama2 iphone
x = Rifky.DataFrame(laptop) #variabel x membuat DataFrame dari library pandas dan akan memanggil variabel laptop. 
print (' Rifky Punya Iphone ' + x) #print hasil dari x

# In[44]: Soal2

import numpy as Rifky #melakukan import numpy sebagai Rifky

matrix_x = Rifky.eye(10) #membuat matrix dengan numpy dengan menggunakan fungsi eye
matrix_x #deklrasikan matrix_x yang telah dibuat

print (matrix_x) #print matrix_x yang telah dibuat dengan 10x10


# In[44]: Soal3

import matplotlib.pyplot as Rifky #import matploblib sebagai Rifky

Rifky.plot([1,1,7,4,0,2,1]) #memberikan nilai plot atau grafik pada Rifky
Rifky.xlabel('Muh Rifky Prananda') #memberikan label pada x
Rifky.ylabel('1174017') #memberikan label pada y
Rifky.show() #print hasil plot berbentuk grafik