# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 11:44:48 2020

@author: Harun
"""

#In[1]: Soal1
import pandas as pd #digunakan untuk import library pandas
harun = pd.read_csv('dataku.csv',sep=';') #digunakan untuk memanggil file csv dan dipisahkan dengan ;
len(harun) #untuk mengetahui jumlah data
print(harun.nama) #digunakan untuk mengetahui kolom nama dari varivel harun
# In[2]: Soal2
import numpy as np #digunakan untuk import library numpy
totumur = np.sum(harun.umur) # menggunakan fungsi sum bawaan dari numpy
print(totumur) #menampilkan isi variabel umur
# In[3]: Soal3
import matplotlib.pyplot as mt #import matploblib
mt.plot([10,20,30,40,50],[1,2,3,4,5]) #digunakan untuk membuat fungsi graph
mt.xlabel("Kecepatan Mobil") #membuat label untuk garis x
mt.ylabel("Gigi") # membuat label untuk y
mt.title("Perubahan Kecepatan Mobil") #membuat judul graph
mt.show() # menampilkan graph