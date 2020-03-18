# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 11:44:48 2020

@author: Ikrima
"""

#In[1]: Soal1
import pandas as pd #digunakan untuk import library pandas
lthn = pd.read_csv('latihan.csv',sep=';') #digunakan untuk memanggil file csv dan dipisahkan dengan ;
len(lthn) #untuk mengetahui jumlah data
print(lthn.jeniskelamin) #digunakan untuk mengetahui kolom nama dari varivel lthn
# In[2]: Soal2
import numpy as np #digunakan untuk import library numpy
pend = np.sum(lthn.pendapatan) # menggunakan fungsi sum bawaan dari numpy
print(pend) #menampilkan isi variabel pendapatan
# In[3]: Soal3
import matplotlib.pyplot as mt #import matploblib
mt.plot([5,15,25,35,45],[1,2,3,4,5]) #digunakan untuk membuat fungsi graph
mt.xlabel("Kinerja") #membuat label untuk garis x
mt.ylabel("Reward") # membuat label untuk y
mt.title("Perubahan Jabatan") #membuat judul graph
mt.show() # menampilkan graph