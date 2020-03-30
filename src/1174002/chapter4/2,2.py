# In[5]: membuat data training dan data testing
# jumlah baris data training sebanyak 450 baris
data_training = data_forest[:450]
# jumlah baris data testing dari hasil pengurangan 523-450
data_testing = data_forest[450:]