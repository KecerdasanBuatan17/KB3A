# In[6]:convert class names into one-hot encoding
# membuat variabel label_encoder dengan isi LabelEncoder
label_encoder = LabelEncoder()
# membuat variabel integer_encoded yang berfungsi untuk mengkonvert variabel classes kedalam bentuk integer
integer_encoded = label_encoder.fit_transform(classes)
