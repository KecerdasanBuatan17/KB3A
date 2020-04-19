# In[7]:then convert integers into one-hot encoding
# membuat variabel onehot_encoder dengan isi OneHotEncoder
onehot_encoder = OneHotEncoder(sparse=False)
# mengisi variabel integer_encoded dengan isi integer_encoded yang telah di convert pada fungsi sebelumnya
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
# mengkonvert variabel integer_encoded kedalam onehot_encoder
onehot_encoder.fit(integer_encoded)
