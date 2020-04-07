# In[]:
import pandas as pd #import package pandas, lalu dialiaskan menjadi pd.
sp = pd.read_csv('StudentsPerformance.csv', delimiter = ',') #membaca file csv dimana data pada file csv dipisahkan oleh koma, lalu ditampung di variable sp.
# In[]:
sp1, sp2 = sp[:450], sp[450:] #membagi data menjadi dua bagian, variable sp1 untuk menampung 450 baris data pertama, variable sp2 untuk menampung 50 baris data terakhir.