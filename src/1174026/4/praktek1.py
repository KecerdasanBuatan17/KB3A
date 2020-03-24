# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 22:15:27 2020

@author: Felix
"""

# In[1]: Import library dari pandas
import pandas as pan
#Membaca file csv menggunakan pandas
data_forest = pan.read_csv('1174026.csv')
# In[2]: untuk melihat jumlah dari baris data yang telah di import
print(len(data_forest))
# In[3]: untuk melihat lima baris pertama data yang telah di import
print(data_forest.head())
# In[4]: untuk mengetahui banyak baris dan kolom dari data yang
# telah di import.
print(data_forest.shape)
#%%
data_450 = data_forest[:450]
#%%
data_50 = data_forest[450:]