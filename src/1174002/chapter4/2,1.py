# In[1]: mengimport librari padas yang di gunakan
# untuk membaca file tex atau csv
import pandas as pd
#membaca file csv menggunakan fungsi read csv dari padas
data_forest = pd.read_csv("D:/Semester6/Kecerdasan_Buatan/ch4/data.csv")
# In[2]: untuk melihat jumlah dari baris data yang telah di import
print(len(data_forest))
# In[3]: untuk melihat lima baris pertama data yang telah di import
print(data_forest.head())
# In[4]: untuk mengetahui banyak baris dan kolom dari data yang
# telah di import.
print(data_forest.shape)
