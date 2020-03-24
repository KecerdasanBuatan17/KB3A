# In[]: Soal 1

import pandas as pd # import package pandas dan dialiaskan sebagai pd
data = {'Nama':['Kadek', 'Diva', 'Krishna', 'Murti'],
        'Umur':[21, 22, 23, 24],
        'Alamat':['Denpasar', 'Badung', 'Gianyar', 'Tabanan']} # buat dictionary yang berisikan list data nama, umur, dan alamat, tampung dictionarynya di variable data
df = pd.DataFrame(data) # konversi dictionary data ke data frame, tampung hasilnya di variable df
print(df[['Nama', 'Alamat']]) # tampilkan data nama dan alamat

# In[]: Soal 2

import numpy as np # import package numpy dan dialiaskan sebagai np
arr = np.array([[ 1, 2, 3], 
                [ 4, 2, 5]]) # buat array menggunakan numpy berisakan 2 baris list data, tampung hasilnya id variable arr
print(arr) # data pada variable arr
print(arr.ndim) # tampilkan dimensi array
print(arr.shape) # tampilkan bentuk array
print(arr.size) # tampilkan jumlah data pada array
print(arr.dtype) # tampilkan tipe data elemen yang disimpan

# In[]: Soal 3

from matplotlib import pyplot as plt # import pyplot dari package matplotlib dan dialiaskan sebagai plt
x = [5, 2, 9, 4, 7] # membuat list berisi nilai untuk sumbu x ditampung di variable x
y = [10, 5, 8, 4, 2] # membuat list berisi nilai untuk sumbu y ditampung di variable y
plt.bar(x,y) # membuat plot bar
plt.show() # menampilkan plot
