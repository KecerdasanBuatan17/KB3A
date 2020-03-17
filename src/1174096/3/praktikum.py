# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 13:19:22 2020

@author: Nico Sembiring
"""
# In[1]:
import pandas as nico #melakukan import pada library pandas sebagai nico
makanan = {"List Nama makanan Nico" : ['Bakso','Mie Ayam','Batagor','Nasi Goreng']} #membuat varibel yang bernama makanan , dan mengisi dataframe nama2 Makanan
makan = nico.DataFrame(makanan) #membuat variabel makan untuk memanggil dataframe makanan
print('Nico Suka ' + makan) #memanggil variabel makan dengan data dari dataframe makanan

# In[2] :
import numpy as nico #melakukan import numpy sebagai nico
matrix = nico.eye(12) #Membuat variabel dengan nama matrix untuk memanggil fungsi eye sebagai matrix identitas dengan jumlah kolom dan baris 12 
print (matrix) #memanggil variabel matrix

# In[3] :
import matplotlib.pyplot as nico #melakukan import pada library matplotlib sebagai nico
nico.plot([1,4,2,4,5,2,1]) #menentukan titik 
nico.ylabel('Nilainya') #mendefinisikan nilai y dengan nama Nilainya
nico.show() #memunculkan grafik

