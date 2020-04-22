# In[14]:rebuild/retrain a model with the best parameters (from the search) and use all data
# membuat variabel model dengan isian librari Sequential
model = Sequential()
# variabel model di tambahkan librari Conv2D tigapuluh dua bit dengan ukuran kernel 3 x 3 dan fungsi penghitungan relu dang menggunakan data train_input
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=np.shape(train_input[0])))
# variabel model di tambahkan dengan lib MaxPooling2D dengan ketentuan ukuran 2 x 2 pixcel
model.add(MaxPooling2D(pool_size=(2, 2)))
# variabel model di tambahkan dengan librari Conv2D 32bit dengan kernel 3 x 3
model.add(Conv2D(32, (3, 3), activation='relu'))
# variabel model di tambahkan dengan lib MaxPooling2D dengan ketentuan ukuran 2 x 2 pixcel
model.add(MaxPooling2D(pool_size=(2, 2)))
# variabel model di tambahkan librari Flatten
model.add(Flatten())
# variabel model di tambahkan librari Dense dengan fungsi tanh
model.add(Dense(128, activation='tanh'))
# variabel model di tambahkan librari dropout untuk memangkas data tree sebesar 50 persen
model.add(Dropout(0.5))
# variabel model di tambahkan librari Dense dengan data dari num_classes dan fungsi softmax
model.add(Dense(num_classes, activation='softmax'))
# mengkompile data model untuk mendapatkan data loss akurasi dan optimasi
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# mencetak variabel model kemudian memunculkan kesimpulan berupa data total parameter, trainable paremeter dan bukan trainable parameter
print(model.summary())
