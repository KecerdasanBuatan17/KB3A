# In[13]:try various model configurations and parameters to find the best
# mengimport librari time 
import time
#membuat variabel result dengan array kosong
results = []
# melakukan looping dengan ketentuan konvolusi 2 dimensi 1 2 
for conv2d_count in [1, 2]:
    # menentukan ukuran besaran fixcel dari data atau konvert 1 fixcel mnjadi data yang berada pada codigan dibawah.
    for dense_size in [128, 256, 512, 1024, 2048]:
        # membuat looping untuk memangkas masing-masing data dengan ketentuan 0 persen 25 persen 50 persen dan 75 persen.
        for dropout in [0.0, 0.25, 0.50, 0.75]:
            # membuat variabel model Sequential
            model = Sequential()
            #membuat looping untuk variabel i dengan jarak dari hasil konvolusi.
            for i in range(conv2d_count):
                # syarat jika i samadengan bobotnya 0
                if i == 0:
                    # menambahkan method add pada variabel model dengan konvolusi 2 dimensi 32 bit didalamnya dan membuat kernel dengan ukuran 3 x 3 dan rumus aktifasi relu dan data shape yang di hitung dari data train.
                    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=np.shape(train_input[0])))
                    # jika tidak
                else:
                    # menambahkan method add pada variabel model dengan konvolusi 2 dimensi 32 bit dengan ukuran kernel 3 x3 dan fungsi aktivasi relu
                    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
                # menambahkan method add pada variabel model dengan isian method  Max pooling berdimensi 2 dengan ukuran fixcel 2 x 2.
                model.add(MaxPooling2D(pool_size=(2, 2)))
            # merubah feature gambar menjadi 1 dimensi vektor
            model.add(Flatten())
            # menambahkan method dense untuk pemadatan data dengan ukuran dense di tentukan dengan rumus fungsi tanh.
            model.add(Dense(dense_size, activation='tanh'))
            # membuat ketentuan jika pemangkasan lebih besar dari 0 persen
            if dropout > 0.0:
                # menambahkan method dropout pada model dengan nilai dari dropout
                model.add(Dropout(dropout))
                # menambahkan method dense dengan fungsi num classs dan rumus softmax
            model.add(Dense(num_classes, activation='softmax'))
            # mongkompile variabel model dengan hasi loss optimasi dan akurasi matrix
            model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
            # melakukan log pada dir 
            log_dir = './logs/conv2d_%d-dense_%d-dropout_%.2f' % (conv2d_count, dense_size, dropout)
            # membuat variabel tensorboard dengan isian dari librari keras dan nilai dari lig dir
            tensorboard = keras.callbacks.TensorBoard(log_dir=log_dir)
            # membuat variabel start dengan isian dari librari time menggunakan method time

            start = time.time()
            # menambahkan method fit pada model dengan data dari train input train output nilai batch nilai epoch verbose nilai 20 persen validation split dan callback dengan nilai tnsorboard.
            model.fit(train_input, train_output, batch_size=32, epochs=10,
                      verbose=0, validation_split=0.2, callbacks=[tensorboard])
            # membuat variabel score dengan nilai evaluasi dari model menggunakan data tes input dan tes output
            score = model.evaluate(test_input, test_output, verbose=2)
            # membuat variabel end 
            end = time.time()
            # membuat variabel elapsed
            elapsed = end - start
            # mencetak hasil perhitungan
            print("Conv2D count: %d, Dense size: %d, Dropout: %.2f - Loss: %.2f, Accuracy: %.2f, Time: %d sec" % (conv2d_count, dense_size, dropout, score[0], score[1], elapsed))
            results.append((conv2d_count, dense_size, dropout, score[0], score[1], elapsed))


